python run_race.py ^
--data_dir ./data/sequence ^
--vocab_file ../blc/vocab.txt ^
--bert_model_src ../blc ^
--do_train ^
--do_test ^
--train_batch_size 16 ^
--dev_batch_size 8 ^
--test_batch_size 8 ^
--num_train_epochs 3.0 ^