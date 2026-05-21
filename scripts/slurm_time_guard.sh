#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: slurm_time_guard.sh [JOB_ID]

Poll a Slurm job and print terminal reminders as walltime runs down.

Environment:
  SLURM_TIME_GUARD_WARN_MINUTES   Comma-separated thresholds. Default: 60,30,15,5,1
  SLURM_TIME_GUARD_INTERVAL       Poll interval in seconds. Default: 60
  SLURM_TIME_GUARD_SQUEUE_TIMEOUT squeue call timeout in seconds. Default: 10
  SLURM_TIME_GUARD_ONCE           If set to 1, print current status once and exit.
  SLURM_TIME_GUARD_HOOK           Optional shell command to run at each warning.
  SLURM_TIME_GUARD_LOG            Optional file path to append warnings/status.

The hook receives JOB_ID, TIME_LEFT_SECONDS, TIME_LEFT, WARN_MINUTES, and WARN_SECONDS.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

JOB_ID="${1:-${SLURM_JOB_ID:-}}"
if [[ -z "$JOB_ID" ]]; then
    echo "slurm_time_guard: JOB_ID argument or SLURM_JOB_ID is required." >&2
    exit 2
fi

WARN_MINUTES="${SLURM_TIME_GUARD_WARN_MINUTES:-60,30,15,5,1}"
INTERVAL="${SLURM_TIME_GUARD_INTERVAL:-60}"
SQUEUE_TIMEOUT="${SLURM_TIME_GUARD_SQUEUE_TIMEOUT:-10}"
ONCE="${SLURM_TIME_GUARD_ONCE:-0}"
HOOK="${SLURM_TIME_GUARD_HOOK:-}"
LOG_FILE="${SLURM_TIME_GUARD_LOG:-}"

parse_time_left() {
    local raw="$1"
    local days=0 hours=0 minutes=0 seconds=0 rest="$raw"

    if [[ "$raw" == "UNLIMITED" || "$raw" == "NOT_SET" || "$raw" == "N/A" ]]; then
        return 1
    fi
    if [[ "$rest" == *-* ]]; then
        days="${rest%%-*}"
        rest="${rest#*-}"
    fi

    IFS=: read -r -a parts <<<"$rest"
    case "${#parts[@]}" in
        3)
            hours="${parts[0]}"
            minutes="${parts[1]}"
            seconds="${parts[2]}"
            ;;
        2)
            minutes="${parts[0]}"
            seconds="${parts[1]}"
            ;;
        1)
            seconds="${parts[0]}"
            ;;
        *)
            return 1
            ;;
    esac

    if ! [[ "$days" =~ ^[0-9]+$ && "$hours" =~ ^[0-9]+$ && "$minutes" =~ ^[0-9]+$ && "$seconds" =~ ^[0-9]+$ ]]; then
        return 1
    fi

    echo $((days * 86400 + hours * 3600 + minutes * 60 + seconds))
}

format_seconds() {
    local total="$1"
    local days=$((total / 86400))
    local hours=$(((total % 86400) / 3600))
    local minutes=$(((total % 3600) / 60))
    local seconds=$((total % 60))

    if (( days > 0 )); then
        printf "%d-%02d:%02d:%02d" "$days" "$hours" "$minutes" "$seconds"
    else
        printf "%02d:%02d:%02d" "$hours" "$minutes" "$seconds"
    fi
}

emit() {
    local message="$1"
    local stamped
    stamped="$(date '+%Y-%m-%d %H:%M:%S') $message"
    echo "$stamped" >&2
    if [[ -n "$LOG_FILE" ]]; then
        mkdir -p "$(dirname "$LOG_FILE")"
        echo "$stamped" >>"$LOG_FILE"
    fi
}

threshold_seconds() {
    local item
    for item in ${WARN_MINUTES//,/ }; do
        item="${item//[[:space:]]/}"
        if [[ -n "$item" && "$item" =~ ^[0-9]+$ ]]; then
            echo $((item * 60))
        fi
    done | sort -rn
}

mapfile -t THRESHOLDS < <(threshold_seconds)
if (( ${#THRESHOLDS[@]} == 0 )); then
    echo "slurm_time_guard: no valid warning thresholds in SLURM_TIME_GUARD_WARN_MINUTES=$WARN_MINUTES" >&2
    exit 2
fi

declare -A WARNED=()
emit "slurm_time_guard: watching job $JOB_ID; warning thresholds: $WARN_MINUTES minutes"

while true; do
    if ! row="$(timeout "$SQUEUE_TIMEOUT" squeue -h -j "$JOB_ID" -o "%L|%M|%T|%P|%j" 2>&1)"; then
        emit "slurm_time_guard: squeue failed for job $JOB_ID: $row"
        [[ "$ONCE" == "1" ]] && exit 1
        sleep "$INTERVAL"
        continue
    fi
    if [[ -z "$row" ]]; then
        emit "slurm_time_guard: job $JOB_ID is no longer in squeue; stopping"
        exit 0
    fi

    IFS='|' read -r time_left elapsed state partition name <<<"$row"
    if ! remaining="$(parse_time_left "$time_left")"; then
        emit "slurm_time_guard: job $JOB_ID state=$state partition=$partition time_left=$time_left elapsed=$elapsed"
        [[ "$ONCE" == "1" ]] && exit 0
        sleep "$INTERVAL"
        continue
    fi

    if [[ "$ONCE" == "1" ]]; then
        emit "slurm_time_guard: job $JOB_ID state=$state partition=$partition remaining=$(format_seconds "$remaining") elapsed=$elapsed name=$name"
        exit 0
    fi

    for threshold in "${THRESHOLDS[@]}"; do
        if (( remaining <= threshold )) && [[ -z "${WARNED[$threshold]:-}" ]]; then
            WARNED[$threshold]=1
            warn_minutes=$((threshold / 60))
            emit "SLURM TIME WARNING: job $JOB_ID has about $(format_seconds "$remaining") left (threshold ${warn_minutes}m). Save checkpoints, patches, logs, and terminal notes now."
            if [[ -n "$HOOK" ]]; then
                JOB_ID="$JOB_ID" \
                TIME_LEFT_SECONDS="$remaining" \
                TIME_LEFT="$(format_seconds "$remaining")" \
                WARN_MINUTES="$warn_minutes" \
                WARN_SECONDS="$threshold" \
                    bash -lc "$HOOK" || emit "slurm_time_guard: warning hook failed at ${warn_minutes}m threshold"
            fi
        fi
    done

    if (( remaining <= 0 )); then
        emit "slurm_time_guard: job $JOB_ID has no remaining walltime; stopping"
        exit 0
    fi

    sleep "$INTERVAL"
done
