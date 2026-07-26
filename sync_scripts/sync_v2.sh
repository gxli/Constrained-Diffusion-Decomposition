#!/usr/bin/env bash
# If invoked via `sh sync_v2.sh ...`, re-exec under bash so arrays/strict mode behave correctly.
if [ -z "${BASH_VERSION:-}" ]; then
  exec bash "$0" "$@"
fi
set -euo pipefail

# Mirror full relative path under ~/proj
BASE_LOCAL_ROOT="$HOME/proj"
if [[ "$PWD" != "$BASE_LOCAL_ROOT"* ]]; then
    echo "ERROR: Current directory must be under $BASE_LOCAL_ROOT"
    exit 1
fi
REL_PATH="${PWD#"$BASE_LOCAL_ROOT"/}"
REMOTE="gxli@100.83.34.55"
LOCAL_RESULTS_DIR="result_local"

# Default version suffix for remote directory (enables versioned parallel workspaces).
SUFFIX=""

SSH_OPTS=(
    -o ConnectTimeout=15
    -o ServerAliveInterval=30
    -o ServerAliveCountMax=3
    -o TCPKeepAlive=yes
)
RSYNC_RSH='ssh -o ConnectTimeout=15 -o ServerAliveInterval=30 -o ServerAliveCountMax=3 -o TCPKeepAlive=yes'
RSYNC_COMMON=(
    -acvz
    --checksum
    --partial
    --timeout=120
    --contimeout=20
    -e "$RSYNC_RSH"
)

# For pull-plots we intentionally skip checksum work and never overwrite existing local files.
RSYNC_PLOT_PULL_COMMON=(
    -avz
    --partial
    --timeout=120
    --contimeout=20
    -e "$RSYNC_RSH"
)

RSYNC_RESUME_FLAG=""
if rsync --help 2>/dev/null | grep -q -- '--append-verify'; then
    RSYNC_RESUME_FLAG="--append-verify"
elif rsync --help 2>/dev/null | grep -q -- '--append'; then
    RSYNC_RESUME_FLAG="--append"
fi

RSYNC_MKPATH_FLAG=""
if rsync --help 2>/dev/null | grep -q -- '--mkpath'; then
    RSYNC_MKPATH_FLAG="--mkpath"
fi

run_rsync_retry() {
    # Retries transient rsync/ssh failures (e.g., status 11 over unstable link).
    local attempts=3
    local n=1
    while true; do
        if rsync "$@"; then
            return 0
        fi
        rc=$?
        if [[ "${rc}" -eq 0 ]]; then
            rc=1
        fi
        if [[ $n -ge $attempts ]]; then
            echo "rsync failed after ${attempts} attempts (exit ${rc})"
            return "$rc"
        fi
        echo "rsync attempt ${n}/${attempts} failed (exit ${rc}), retrying..."
        sleep $((2 * n))
        n=$((n + 1))
    done
}


usage() {
    echo "----------------------------------------------------------------"
    echo "Sync Script for: $REL_PATH"
    echo "----------------------------------------------------------------"
    echo "Usage: $0 {push|push-preview|pull|pull-plots|pull_plots|pull-plot|pull_plot|pull-all} [suffix]"
    echo ""
    echo "Arguments:"
    echo "  suffix  -> Remote directory version suffix (e.g. _v2)."
    echo "             Remote path becomes .../proj/<project>_v2/"
    echo ""
    echo "Commands:"
    echo "  push        -> Local to Remote mirror APPLY (ignores sessions/, results/, result_local/)"
    echo "  push-preview-> Local to Remote mirror PREVIEW (dry-run, ignores sessions/, results/, result_local/)"
    echo "  pull        -> Remote outputs/ to Local $LOCAL_RESULTS_DIR/ (all files)"
    echo "  pull-plots  -> Remote results/ to Local $LOCAL_RESULTS_DIR/ (only .html/.png/.jpg/.pdf/.svg)"
    echo "                 skips files that already exist locally (no checksum)"
    echo "  pull_plots  -> Alias of pull-plots"
    echo "  pull-plot   -> Alias of pull-plots"
    echo "  pull_plot   -> Alias of pull-plots"
    echo "  pull-all    -> Remote project to Local current directory (legacy full pull)"
    echo "----------------------------------------------------------------"
    exit 1
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
fi

if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
    usage
fi

# Parse optional 2nd arg: suffix
if [[ -n "${2:-}" ]]; then
    SUFFIX="$2"
fi

# Resolve remote HOME explicitly (avoid '~' path ambiguity across shells/hosts).
echo "Resolving remote HOME on $REMOTE ..."
REMOTE_HOME="$(ssh "${SSH_OPTS[@]}" "$REMOTE" 'printf %s "$HOME"')"
if [[ -z "${REMOTE_HOME}" ]]; then
    echo "ERROR: Could not resolve remote HOME for $REMOTE"
    exit 1
fi
echo "Remote HOME: $REMOTE_HOME"

# Remote destination with version suffix
VERSIONED_REL_PATH="${REL_PATH}${SUFFIX}"
REMOTE_DEST="${REMOTE_HOME}/proj/${VERSIONED_REL_PATH}/"
REMOTE_OUTPUTS_DEST="${REMOTE_HOME}/proj/${VERSIONED_REL_PATH}/outputs/"

case "$1" in
    push)
        echo "Applying safe mirror push to $REMOTE:$REMOTE_DEST (preserve excluded dirs)"
        ssh "${SSH_OPTS[@]}" "$REMOTE" "mkdir -p \"$REMOTE_HOME/proj/$VERSIONED_REL_PATH\""
        # Do not use --append/--append-verify for mutable project files. Those
        # flags are for resuming large downloads and can corrupt or skip edits
        # to existing files under src/ and configs/.
        run_rsync_retry "${RSYNC_COMMON[@]}" \
            --delete --force \
            --exclude '.git' \
            --exclude 'result_local/' \
            --exclude 'results/' \
            --exclude 'sessions/' \
            --exclude 'outputs/' \
            --exclude 'build/' \
            --exclude '*.egg-info/' \
            --exclude '__pycache__/' \
            --exclude '*.pyc' \
            --include='local_tests/' \
            --include='local_tests/**.py' \
            --exclude='local_tests/*' \
            ./ "$REMOTE:$REMOTE_DEST"
        ;;
    push-preview)
        echo "Previewing safe mirror push (dry-run) to $REMOTE:$REMOTE_DEST (preserve excluded dirs)"
        ssh "${SSH_OPTS[@]}" "$REMOTE" "mkdir -p \"$REMOTE_HOME/proj/$VERSIONED_REL_PATH\""
        run_rsync_retry "${RSYNC_COMMON[@]}" \
            --dry-run \
            --delete --force \
            --exclude '.git' \
            --exclude 'result_local/' \
            --exclude 'results/' \
            --exclude 'sessions/' \
            --exclude 'outputs/' \
            --exclude 'build/' \
            --exclude '*.egg-info/' \
            --exclude '__pycache__/' \
            --include='local_tests/' \
            --include='local_tests/**.py' \
            --exclude='local_tests/*' \
            --exclude '*.pyc' \
            ./ "$REMOTE:$REMOTE_DEST"
        ;;
    pull)
        echo "Pulling remote outputs from $REMOTE:$REMOTE_OUTPUTS_DEST to ./$LOCAL_RESULTS_DIR/"
        mkdir -p "$LOCAL_RESULTS_DIR"
        run_rsync_retry "${RSYNC_COMMON[@]}" \
            ${RSYNC_RESUME_FLAG:+$RSYNC_RESUME_FLAG} \
            "$REMOTE:$REMOTE_OUTPUTS_DEST" "$LOCAL_RESULTS_DIR"/
        ;;
    pull-plots|pull_plots|pull-plot|pull_plot)
        echo "Pulling plot files (.html and images) from remote results/ to ./$LOCAL_RESULTS_DIR/ (skip existing local files)"
        mkdir -p "$LOCAL_RESULTS_DIR"
        d="$REMOTE_HOME/proj/$VERSIONED_REL_PATH/results/"
        local_dst="$LOCAL_RESULTS_DIR/results/"
        echo "  trying: $REMOTE:$d -> ./$local_dst"
        if ssh "${SSH_OPTS[@]}" "$REMOTE" "test -d \"$d\""; then
            mkdir -p "$local_dst"
            # Precreate common nested plot destinations to avoid move_file errors on some rsync builds.
            mkdir -p "$local_dst/dashboard" "$local_dst/plots/session_dashboards" "$local_dst/dashboards" "$local_dst/plots"
            # Resume flags (--append/--append-verify) can interact badly with filtered tree pulls.
            # Use plain transfer for pull-plots stability.
            run_rsync_retry "${RSYNC_PLOT_PULL_COMMON[@]}" \
                ${RSYNC_MKPATH_FLAG:+$RSYNC_MKPATH_FLAG} \
                --ignore-existing \
                --include='*/' \
                --include='*.html' --include='*.htm' \
                --include='*.png'  --include='*.jpg' \
                --include='*.jpeg' --include='*.pdf' \
                --include='*.svg' \
                --exclude='*' "$REMOTE:$d" "$local_dst"
        else
            echo "    skipped (missing): $d"
        fi
        ;;
    pull-all)
        echo "Pulling all remote contents from $REMOTE:$REMOTE_DEST to current directory"
        run_rsync_retry "${RSYNC_COMMON[@]}" \
            ${RSYNC_RESUME_FLAG:+$RSYNC_RESUME_FLAG} \
            --exclude '.git' "$REMOTE:$REMOTE_DEST" ./
        ;;
    *)
        usage
        ;;
esac
