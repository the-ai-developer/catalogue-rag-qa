#!/bin/sh
# Fix ownership on the volumes this container *owns*, then drop privileges.
#
# Ownership contract:
#   /data/faiss        written by this container  -> chown to APP_UID
#   /app/ml/checkpoints  written by the trainer  -> chown to APP_UID
#   /data/assets        written by the api image -> READ ONLY here
#
# The assets volume is shared with the api container, which runs as a different
# uid. Chowning it from here made the api lose write access (photo upload failed
# with permission denied), so it is deliberately not touched.
set -eu

APP_UID="${APP_UID:-10001}"
APP_GID="${APP_GID:-10001}"

OWNED="${FAISS_INDEX_DIR:-/data/faiss}"
case "${PROJECTION_PATH:-}" in
  /*) OWNED="$OWNED $(dirname "${PROJECTION_PATH}")" ;;
esac

for dir in $OWNED; do
  if [ -d "$dir" ]; then
    chown -R "${APP_UID}:${APP_GID}" "$dir" 2>/dev/null || \
      echo "warn: could not chown $dir (read-only mount?)" >&2
  fi
done

# Readability check for the shared volume, so a misconfigured ASSET_ROOT is
# obvious in the logs instead of showing up as a failed image embedding.
if [ -n "${ASSET_ROOT:-}" ] && [ ! -r "${ASSET_ROOT}" ]; then
  echo "warn: ASSET_ROOT=${ASSET_ROOT} is not readable by uid ${APP_UID};" >&2
  echo "warn: image embedding will fail. Check the assets volume." >&2
fi

if [ "$(id -u)" = "0" ]; then
  # setpriv ships in util-linux, so it is present in every debian-based image;
  # su-exec and gosu are optional extras.
  if command -v setpriv >/dev/null 2>&1; then
    exec setpriv --reuid "${APP_UID}" --regid "${APP_GID}" --clear-groups -- "$@"
  elif command -v su-exec >/dev/null 2>&1; then
    exec su-exec "${APP_UID}:${APP_GID}" "$@"
  elif command -v gosu >/dev/null 2>&1; then
    exec gosu "${APP_UID}:${APP_GID}" "$@"
  else
    echo "warn: no setpriv/su-exec/gosu; running as root" >&2
  fi
fi
exec "$@"
