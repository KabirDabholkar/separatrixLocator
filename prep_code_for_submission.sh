
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
  ./ ../separatrixLocatorneurips
