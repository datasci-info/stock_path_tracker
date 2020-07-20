tmux new-session \; \
  send-keys 'htop' ENTER \; \
split-window -v -p 75 \; \
  send-keys 'sudo iotop' ENTER \; \
split-window -v -p 75\; \
  send-keys 'git log --graph --all --decorate --oneline' ENTER \; \
split-window -h \; \
  send-keys 'cd /data/dataset/' ENTER \; \
split-window -v -p 30\; \
  send-keys 'conda activate stock_path_tracker; jupyter-lab --port 8090' ENTER \; \
select-pane  -t 2 \; \
split-window -v -p 50 \; \
  send-keys 'conda activate stock_path_tracker' ENTER \; \
