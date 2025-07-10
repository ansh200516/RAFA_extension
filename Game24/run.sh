#!/bin/bash

# --- Configuration ---
# You can change these variables to configure the run.

# Core arguments
BACKEND="gpt-4.1-nano"
TASK="game24"
TASK_FILE_PATH="24.csv"
TASK_START_INDEX=900
TASK_END_INDEX=915
PLANNING="summary"
K_MEMORY=1
FEEDBACK_ENABLED=true # Set to true to enable feedback, false to disable
SUMMARY_SIZE_PERCENTAGE=70 # Percentage of the input text to use for the summary

# Tree of Thoughts (ToT) arguments
PROMPT_SAMPLE="standard"
N_GENERATE_SAMPLE=10
METHOD_GENERATE="propose"
METHOD_EVALUATE="value"
N_EVALUATE_SAMPLE=3
METHOD_SELECT="greedy"
N_SELECT_SAMPLE=2

# --- Argument Construction ---
ARGS=(
    --backend "$BACKEND"
    --task "$TASK"
    --task_file_path "$TASK_FILE_PATH"
    --task_start_index "$TASK_START_INDEX"
    --task_end_index "$TASK_END_INDEX"
    --prompt_sample "$PROMPT_SAMPLE"
    --n_generate_sample "$N_GENERATE_SAMPLE"
    --method_generate "$METHOD_GENERATE"
    --method_evaluate "$METHOD_EVALUATE"
    --n_evaluate_sample "$N_EVALUATE_SAMPLE"
    --method_select "$METHOD_SELECT"
    --n_select_sample "$N_SELECT_SAMPLE"
    --planning "$PLANNING"
)

if [ "$PLANNING" = "summary" ]; then
    ARGS+=(--summary_size_percentage "$SUMMARY_SIZE_PERCENTAGE")
fi

if [ "$PLANNING" = "prevk" ]; then
    ARGS+=(--k_memory "$K_MEMORY")
fi

if [ "$FEEDBACK_ENABLED" = true ]; then
    ARGS+=(--feedback)
fi

# --- Print Configuration ---
echo "Starting Game24 Run"
echo "================================================"
echo "Configuration:"
echo "  Backend:            $BACKEND"
echo "  Task:               $TASK"
echo "  Task File:          $TASK_FILE_PATH"
echo "  Indices:            $TASK_START_INDEX-$TASK_END_INDEX"
echo "  Planning Strategy:  $PLANNING"
echo "  Feedback:           $FEEDBACK_ENABLED"
if [ "$PLANNING" = "summary" ]; then
    echo "  Summary Size %:     $SUMMARY_SIZE_PERCENTAGE"
fi
if [ "$PLANNING" = "prevk" ]; then
    echo "  K Memory:           $K_MEMORY"
fi
echo
echo "Tree of Thoughts Parameters:"
echo "  Prompt Sample:      $PROMPT_SAMPLE"
echo "  Generate Samples:   $N_GENERATE_SAMPLE"
echo "  Generate Method:    $METHOD_GENERATE"
echo "  Evaluate Samples:   $N_EVALUATE_SAMPLE"
echo "  Evaluate Method:    $METHOD_EVALUATE"
echo "  Select Samples:     $N_SELECT_SAMPLE"
echo "  Select Method:      $METHOD_SELECT"
echo "================================================"
echo

# --- Execute ---
echo "Running command:"
echo "python run.py ${ARGS[*]}"
echo "------------------------------------------------"
echo

# Change to the directory of the script to ensure relative paths work
cd "$(dirname "$0")" || exit

python run.py "${ARGS[@]}" 