python train_test.py --root /data/argoverse_data/v2/train/train_test_0/ --train_batch_size 2 --val_batch_size 2 --test_batch_size 2 --devices 1 --dataset argoverse_v2 --num_historical_steps 50 --num_future_steps 60 --num_recurrent_steps 3 --pl2pl_radius 150 --time_span 10 --pl2a_radius 50 --a2a_radius 50 --num_t2m_steps 30 --pl2m_radius 150 --a2m_radius 150