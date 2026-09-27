// Command server runs the Catalogue AI API: HTTP + background workers.
package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"strings"
	"syscall"
	"time"

	"catalogue-ai/services/api/internal/config"
	"catalogue-ai/services/api/internal/db"
	"catalogue-ai/services/api/internal/httpapi"
	"catalogue-ai/services/api/internal/modelserver"
	"catalogue-ai/services/api/internal/store"
	"catalogue-ai/services/api/internal/workers"

	"github.com/jackc/pgx/v5/pgxpool"
)

func main() {
	// `-healthcheck` exists because the production image is distroless: there is
	// no shell, no curl and no wget for a container HEALTHCHECK to call.
	var healthcheck bool
	flag.BoolVar(&healthcheck, "healthcheck", false,
		"probe HTTP_ADDR/readyz and exit 0 when serving, 1 otherwise")
	flag.Parse()
	if healthcheck {
		os.Exit(runHealthcheck())
	}

	cfg, err := config.Load()
	if err != nil {
		slog.Error("config", "err", err)
		os.Exit(1)
	}
	slog.SetDefault(slog.New(slog.NewJSONHandler(os.Stdout, &slog.HandlerOptions{
		Level: parseLevel(cfg.LogLevel)})))

	ctx, stop := signal.NotifyContext(context.Background(),
		syscall.SIGINT, syscall.SIGTERM)
	defer stop()

	pool, err := connectWithRetry(ctx, cfg.DatabaseURL, cfg.DBRetryBudget)
	if err != nil {
		slog.Error("db connect", "err", err)
		os.Exit(1)
	}
	defer pool.Close()

	if err := db.Migrate(ctx, pool, cfg.MigrationsDir); err != nil {
		slog.Error("migrate", "err", err)
		os.Exit(1)
	}

	st := store.New(pool)
	ms := modelserver.New(cfg.ModelServerURL, cfg.ModelTimeoutShort, cfg.ModelTimeoutLong).
		WithSecret(cfg.ModelSharedSecret)
	keys := httpapi.Chain{httpapi.NewCached(st, cfg.AuthCacheTTL), httpapi.BootstrapKeys(cfg)}

	// Fail loudly at boot rather than on the first upload: a root-owned assets
	// volume against a non-root container is invisible until someone tries to
	// add a product photo.
	if err := checkAssetDir(cfg.AssetStorageDir); err != nil {
		slog.Warn("asset storage is not writable; photo upload will fail", "err", err)
	}

	srv := httpapi.New(st, ms, cfg, keys)
	worker := &workers.Worker{Store: st, MS: ms, Cfg: cfg}
	worker.Run(ctx)

	httpSrv := &http.Server{
		Addr:              cfg.HTTPAddr,
		Handler:           srv.Router(),
		ReadHeaderTimeout: 10 * time.Second,
		ReadTimeout:       60 * time.Second,
		WriteTimeout:      120 * time.Second, // /qa/ask composes against the model server
		IdleTimeout:       120 * time.Second,
		MaxHeaderBytes:    1 << 20,
	}
	go func() {
		slog.Info("api listening", "addr", cfg.HTTPAddr)
		if err := httpSrv.ListenAndServe(); err != nil &&
			!errors.Is(err, http.ErrServerClosed) {
			slog.Error("http", "err", err)
			stop()
		}
	}()

	<-ctx.Done()
	slog.Info("shutting down")
	shutdownCtx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()
	if err := httpSrv.Shutdown(shutdownCtx); err != nil {
		slog.Error("shutdown", "err", err)
	}
}

// connectWithRetry waits for Postgres instead of exiting on the first refusal.
//
// Compose and Kubernetes both gate the api on a health check, but a health
// check can report ready marginally before the server accepts TCP connections.
// Exiting there turned a two-second startup race into a permanently dead
// container, so the api retries for up to budget (DB_RETRY_BUDGET, default 2m)
// and then gives up loudly.
func connectWithRetry(ctx context.Context, url string, budget time.Duration) (*pgxpool.Pool, error) {
	deadline := time.Now().Add(budget)
	attempt := 0
	for {
		attempt++
		pool, err := db.Connect(ctx, url)
		if err == nil {
			if attempt > 1 {
				slog.Info("database reachable", "attempts", attempt)
			}
			return pool, nil
		}
		if ctx.Err() != nil {
			return nil, ctx.Err()
		}
		if time.Now().After(deadline) {
			return nil, err
		}
		wait := time.Duration(attempt) * time.Second
		if wait > 5*time.Second {
			wait = 5 * time.Second
		}
		slog.Warn("waiting for database", "attempt", attempt, "retry_in", wait, "err", err)
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		case <-time.After(wait):
		}
	}
}

// checkAssetDir verifies the api can actually write uploads. It creates a
// throwaway probe file rather than trusting the directory mode, because a
// read-only or root-owned mount only shows up on write.
func checkAssetDir(dir string) error {
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return fmt.Errorf("cannot create %s: %w", dir, err)
	}
	probe, err := os.CreateTemp(dir, ".write-probe-*")
	if err != nil {
		return fmt.Errorf("cannot write in %s: %w", dir, err)
	}
	name := probe.Name()
	probe.Close()
	return os.Remove(name)
}

// runHealthcheck probes /healthz (liveness: the process is up and routing) and
// not /readyz, which also requires Postgres and the model server — a container
// HEALTHCHECK should not flap because a dependency is briefly down.
func runHealthcheck() int {
	addr := os.Getenv("HTTP_ADDR")
	if addr == "" {
		addr = ":8080"
	}
	if !strings.HasPrefix(addr, "http://") && !strings.HasPrefix(addr, "https://") {
		addr = "http://" + addr
	}
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, addr+"/healthz", nil)
	if err != nil {
		return 1
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return 1
	}
	defer resp.Body.Close()
	_, _ = io.Copy(io.Discard, resp.Body)
	if resp.StatusCode != http.StatusOK {
		return 1
	}
	return 0
}

func parseLevel(s string) slog.Level {
	switch s {
	case "debug":
		return slog.LevelDebug
	case "warn":
		return slog.LevelWarn
	case "error":
		return slog.LevelError
	default:
		return slog.LevelInfo
	}
}
