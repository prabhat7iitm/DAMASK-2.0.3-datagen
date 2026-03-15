#!/bin/bash
set -euo pipefail  # Exit on error, undefined variables, and pipeline errors

# === Start Timer ===
SECONDS=0  # Tracks total script execution time

# === Configuration ===
i="$1"  # Task ID from parallel (ensure it's an integer)
BASE_DIR="/home/Crystal_Elasticity/Cu" ### Check before run
DEST_DIR="$BASE_DIR/extract_data"
DIR="$BASE_DIR/d_$i"

# === Validate Input ===
if ! [[ "$i" =~ ^[0-9]+$ ]]; then
  echo "❌ Invalid task ID: $i. Must be an integer."
  exit 1
fi

# === Load Required Modules ===
module load damask2 >/dev/null 2>&1 || { echo "❌ Failed to load DAMASK module"; exit 1; }

# === Create Directories ===
mkdir -p "$DEST_DIR" || { echo "❌ Failed to create $DEST_DIR"; exit 1; }
mkdir -p "$DIR" || { echo "❌ Failed to create $DIR"; exit 1; }

# === Copy Essential Files ===
copy_files=(
  "$BASE_DIR/preprocessing.sh"
  "$BASE_DIR/spectral_solver.sh"
  "$BASE_DIR/postprocessing.sh"
  "$BASE_DIR"/*.load
  "$BASE_DIR"/*.config
)
for file in "${copy_files[@]}"; do
  cp "$file" "$DIR/" 2>/dev/null || echo "⚠️ Could not copy $file to $DIR/"
done

# === Generate material.config ===
echo "{d_$i.geom}" >> "$DIR/material.config" || { echo "❌ Failed to write material.config"; exit 1; }

# === Create Preprocessing Script ===
cat <<EOF > "$DIR/preprocessing.sh"
#!/bin/bash
module load anaconda3 < /dev/null

# Generate seeds and geometry       # N>> Number of grains    
seeds_fromRandom -N 64 --grid 100 100 1 > "d_$i.seeds" || { echo "❌ seeds_fromRandom failed"; exit 1; }
geom_fromVoronoiTessellation -g 100 100 1 < "d_$i.seeds" > "d_$i.geom" || { echo "❌ geom_fromVoronoiTessellation failed"; exit 1; }
geom_check "d_$i.geom" || { echo "❌ geom_check failed"; exit 1; }
EOF

chmod +x "$DIR/preprocessing.sh"


# === Run Simulations ===
cd "$DIR" || { echo "❌ Failed to cd into $DIR"; exit 1; }

# Step 1: Preprocessing
echo "=== Running preprocessing for d_$i ==="
bash preprocessing.sh > preprocessing.log 2>&1 || { echo "❌ Preprocessing failed"; exit 1; }


# Step 2: Spectral Solver
echo "=== Running spectral solver for d_$i ==="
bash spectral_solver.sh > solver.log 2>&1 || { echo "❌ Spectral solver failed"; exit 1; }

# Step 3: Postprocessing
echo "=== Running postprocessing for d_$i ==="
bash postprocessing.sh > postprocessing.log 2>&1 || { echo "❌ Postprocessing failed"; exit 1; }

# === Copy Results ===
copy_results=(
  *.geom
  postProc/*.txt
)

for result in "${copy_results[@]}"; do
  cp "$result" "$DEST_DIR/" 2>/dev/null || echo "⚠️ Failed to copy $result to $DEST_DIR/"
done

# === Cleanup ===
if [[ -f "$DEST_DIR/d_$i.geom" ]]; then
  echo "✅ Success: Results copied. Deleting $DIR..."
  rm -rf "$DIR"
else
  echo "❌ Critical Error: Essential files missing. Preserving $DIR for debugging."
  exit 1
fi

# === Report Total Runtime ===
duration=$SECONDS
echo "⏱️ Total runtime: $(($duration / 60)) minutes and $(($duration % 60)) seconds."
