
rsync -av \
  --exclude 'results/' \
  --exclude 'outputs/' \
  --exclude 'RNN_ALM_gating/' \
  --exclude 's4/' \
  --exclude '.git/' \
  --exclude '.venv/' \
  --exclude '__pycache__/' \
  --exclude '.idea/' \
  --exclude 'multirun/' \
  --exclude '.gitmodules' \
  --exclude 'tmux-buffer.txt' \
  --exclude 'run_main.sh' \
  --exclude 'prep_code_for_submission.sh' \
  --exclude '.pytest_cache' \
  ./ ../separatrixLocatorneurips

#find ../separatrixLocatorneurips -type f \( ! -name "*.png" ! -name "*.jpg" ! -name "*.pdf" \) \
#  -exec sed -i -E 's/(Kabir|Dabholkar)/anon/Ig' {} +
find ../separatrixLocatorneurips -type f \( -name "*.py" -o -name "*.txt" -o -name "*.gitmodules" -o -name "*.txt-E" -o -name "*.yaml" -o -name "*.yml" -o -name "*.md" \) \
  -exec sed -i '' -E 's/[Kk]abir[dD]?|[Dd]abholkar/anon/g' {} +
#-exec sed -i -E 's/(Kabir|Dabholkar)/anon/Ig' {} +
# macOS-safe, recursive replace


find ../separatrixLocatorneurips -name "*-E" -type f -delete
