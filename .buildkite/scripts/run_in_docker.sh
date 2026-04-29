#!/bin/bash
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#
# .buildkite/run_in_docker.sh
# ---------------------------

# Exit on error, exit on unset variable, fail on pipe errors.
set -euo pipefail

if [ "$#" -eq 0 ]; then
  echo "ERROR: Usage: $0 <command_and_args_to_run_in_docker...>"
  exit 1
fi

declare -a BENCHMARK_DOCKER_ARGS=()

# Check if the serialized string exists and is not empty.
if [ -n "${BENCHMARK_DOCKER_ARGS_STR:-}" ]; then
  mapfile -t BENCHMARK_DOCKER_ARGS <<< "${BENCHMARK_DOCKER_ARGS_STR}"
fi
printf "[INFO] %s = %s\n" "BENCHMARK_DOCKER_ARGS" "${BENCHMARK_DOCKER_ARGS[*]}"

# TODO(Qiliang Cui): This is temp solution to mitigate the docker image
#     not cleaned issue when migrating benchmark to buildkite.
docker rm -f vllm-tpu || true

# Environment variables for docker run
ENV_VARS=(
  -e TEST_MODEL="${TEST_MODEL:-}"
  -e MINIMUM_ACCURACY_THRESHOLD="${MINIMUM_ACCURACY_THRESHOLD:-}"
  -e MINIMUM_THROUGHPUT_THRESHOLD="${MINIMUM_THROUGHPUT_THRESHOLD:-}"
  -e TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-}"
  -e TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-}"
  -e INPUT_LEN="${INPUT_LEN:-}"
  -e OUTPUT_LEN="${OUTPUT_LEN:-}"
  -e PREFIX_LEN="${PREFIX_LEN:-}"
  -e MAX_MODEL_LEN="${MAX_MODEL_LEN:-}"
  -e MAX_NUM_SEQS="${MAX_NUM_SEQS:-}"
  -e MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-}"
)

if [ -z "${MODEL_IMPL_TYPE:-}" ]; then
    MODEL_IMPL_TYPE=flax_nnx
fi

IMAGE_NAME='vllm-tpu'
FULL_IMAGE_TAG="${IMAGE_NAME}:${BUILDKITE_COMMIT}"
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
# Source the environment setup script
# shellcheck disable=SC1091
source "$SCRIPT_DIR/setup_docker_env.sh"
setup_environment $IMAGE_NAME

TEST_SUITE_VARS=(
  -e BUILDKITE_ANALYTICS_TOKEN="${BUILDKITE_ANALYTICS_TOKEN:-}"
  -e BUILDKITE_BUILD_ID="${BUILDKITE_BUILD_ID:-}"
  -e BUILDKITE_BUILD_NUMBER="${BUILDKITE_BUILD_NUMBER:-}"
  -e BUILDKITE_JOB_ID="${BUILDKITE_JOB_ID:-}"
  -e BUILDKITE_BRANCH="${BUILDKITE_BRANCH:-}"
  -e BUILDKITE_COMMIT="${BUILDKITE_COMMIT:-}"
  -e BUILDKITE_MESSAGE="${BUILDKITE_MESSAGE:-}"
  -e BUILDKITE_BUILD_URL="${BUILDKITE_BUILD_URL:-}"
)

DOCKER_HF_HOME="/tmp/hf_home"

# Try to cache HF models
persist_cache_dir="/mnt/disks/persist/models"

if ( mkdir -p "$persist_cache_dir" ); then
  LOCAL_HF_HOME="$persist_cache_dir"
else
  echo "Error: Failed to create $persist_cache_dir"
  exit 1
fi

# Try to cache the JAX compilations.
GCS_CACHE_BASE="gs://ullm-ci-cache/jax_cache"
echo "[INFO] Probing JAX version from docker image..."
JAX_VERSION=$(docker run --rm "$FULL_IMAGE_TAG" python3 -c "import jax; print(jax.__version__)")
echo "[INFO] Detected JAX Version: ${JAX_VERSION}"

# Centralized GCS cache path
CACHE_NAMESPACE="jax${JAX_VERSION}_tpu${TPU_VERSION:-unknown}"
FINAL_CACHE_PATH="${GCS_CACHE_BASE}/${CACHE_NAMESPACE}"

LOCAL_JAX_CACHE_DIR="/tmp/tpu_jax_cache/${CACHE_NAMESPACE}"
mkdir -p "$LOCAL_JAX_CACHE_DIR"
echo "[INFO] Pulling JAX Cache from GCS to local directory..."
gsutil -m rsync -r "$FINAL_CACHE_PATH" "$LOCAL_JAX_CACHE_DIR" || true

# ==========================================
# 2. XLA Dump Toggle Logic
# ==========================================
DUMP_VOL_ARGS=()
DUMP_ENV_ARGS=()
LOCAL_XLA_DUMP_DIR="$(pwd)/xla_dump"

if [[ "${ENABLE_XLA_DUMP:-0}" == "1" ]]; then
  echo "[INFO] XLA Dump is ENABLED. Logs will be saved to ${LOCAL_XLA_DUMP_DIR}"
  mkdir -p "$LOCAL_XLA_DUMP_DIR"
  DUMP_VOL_ARGS=( -v "${LOCAL_XLA_DUMP_DIR}:/tmp/xla_dump" )
  DUMP_ENV_ARGS=( -e XLA_FLAGS="--xla_dump_to=/tmp/xla_dump --xla_dump_hlo_as_text" )
fi

# ==========================================
# 3. Run Docker Container
# ==========================================
set +e # Temporarily disable exit on error to capture exit code

# Use LOCAL_JAX_CACHE_DIR for cache
docker run \
  --privileged \
  --net host \
  --shm-size=16G \
  --rm \
  -v "$LOCAL_HF_HOME":"$DOCKER_HF_HOME" \
  -v "$LOCAL_JAX_CACHE_DIR":"$LOCAL_JAX_CACHE_DIR" \
  "${DUMP_VOL_ARGS[@]}" \
  "${ENV_VARS[@]}" \
  "${TEST_SUITE_VARS[@]}" \
  -e HF_HOME="$DOCKER_HF_HOME" \
  -e MODEL_IMPL_TYPE="$MODEL_IMPL_TYPE" \
  -e HF_TOKEN="$HF_TOKEN" \
  -e VLLM_XLA_CACHE_PATH="$LOCAL_JAX_CACHE_DIR" \
  -e VLLM_XLA_CHECK_RECOMPILATION=1 \
  -e JAX_LOG_COMPILES=1 \
  -e PYTHONHASHSEED=0 \
  -e JAX_COMPILATION_CACHE_DIR="$LOCAL_JAX_CACHE_DIR" \
  "${DUMP_ENV_ARGS[@]}" \
  ${QUANTIZATION:+-e QUANTIZATION="$QUANTIZATION"} \
  ${NEW_MODEL_DESIGN:+-e NEW_MODEL_DESIGN="$NEW_MODEL_DESIGN"} \
  ${USE_V6E8_QUEUE:+-e USE_V6E8_QUEUE="$USE_V6E8_QUEUE"} \
  ${TPU_VERSION:+-e TPU_VERSION="$TPU_VERSION"} \
  ${SKIP_ACCURACY_TESTS:+-e SKIP_ACCURACY_TESTS="$SKIP_ACCURACY_TESTS"} \
  ${VLLM_MLA_DISABLE:+-e VLLM_MLA_DISABLE="$VLLM_MLA_DISABLE"} \
  "${BENCHMARK_DOCKER_ARGS[@]}" \
  "$FULL_IMAGE_TAG" \
  "$@" 
DOCKER_EXIT_CODE=$?

set -e

# ==========================================
# 4. Post-Docker Actions
# ==========================================
echo "[INFO] Docker finished with exit code ${DOCKER_EXIT_CODE}."

# upload artifacts if ENABLE_XLA_DUMP
if [[ "${ENABLE_XLA_DUMP:-0}" == "1" ]] && command -v buildkite-agent &> /dev/null; then
    echo "[INFO] Uploading XLA dumps to BuildKite Artifacts..."
    buildkite-agent artifact upload "${LOCAL_XLA_DUMP_DIR}/**/*" || echo "[WARNING] Artifact upload failed."
fi

echo "[INFO] Syncing local JAX Cache back to GCS..."
gsutil -m rsync -r "$LOCAL_JAX_CACHE_DIR" "$FINAL_CACHE_PATH" || true

exit $DOCKER_EXIT_CODE