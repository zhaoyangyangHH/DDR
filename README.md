# Rescue Conversations from Dead-ends: Efficient Exploration for Task-oriented Dialogue Policy Optimization<br />


This document describes how to run the simulation of DDR Agent.


## DATA

All the data is located in the folder _./src/deep_dialog/data_domains_, which includes the movie, restaurant, and taxi domains, following the paper 'Dialogue Environments are Different from Games: Investigating Variants of Deep Q-Networks for Dialogue Policy.' Additionally, the platform is based on the paper 'Microsoft Dialogue Challenge: Building End-to-End Task-Completion Dialogue Systems.

__Knowledge Bases__ <br />
movie domain: movie_kb.1k.p <br />
resturant domain: restaurant.kb.nondup.v1.p <br />
taxi domain: taxi.kb.v2.nondup.p <br />

__User Goals__ <br />
movie domain: user_goals_first_turn_template.part.movie.v1.p <br />
resturant domain: user_goals_first.v1.p <br />
taxi domain: user_goals_first.v4.p <br />

We also built 128 user goals for each domain to facilitate small-scale quick evaluations  <br />

movie domain: user_goals_first_turn_template.part.movie.v1.p <br />
resturant domain: goals_128.p <br />
taxi domain: goals_128.p <br />

__NLG Rule Template__ <br />
dia_act_nl_pairs.v6.json  <br />

__Dialog Act Intent__ <br />
dia_acts.txt <br />

__Dialog Act Slot__ <br />
slot_set.txt <br />


## Parameter

We have introduced those parameters in the code.

## Running Dialogue Agents

Movie: bash movie.sh   <br />

for ((i=1;i<=10;i=i+1)) <br />
do <br />
  let "seed=$i*100" <br />
	save_path="./movie/$i" <br />
	mkdir -p $save_path <br />
<br />
	python run.py \ <br />
	--num $i \ <br />
	--torch_seed $seed \ <br />
	--agt 9 \ <br />
	--usr 1 \ <br />
	--max_turn 30 \ <br />
	--kb_path ./deep_dialog/data_movie/movie_kb.1k.p \ <br />
	--goal_file_path ./deep_dialog/data_movie/user_goals_first_turn_template.part.movie.v1.p \ <br />
	--slot_set ./deep_dialog/data_movie/slot_set.txt \ <br />
	--act_set ./deep_dialog/data_movie/dia_acts.txt \ <br />
	--dict_path ./deep_dialog/data_movie/slot_dict.v1.p \ <br />
	--nlg_model_path ./deep_dialog/models/nlg/movie/lstm_tanh_[1533529279.91]_87_99_199_0.988.p \ <br />
	--nlu_model_path ./deep_dialog/models/nlu/movie/lstm_[1533588045.3]_38_38_240_0.998.p \ <br />
	--diaact_nl_pairs ./deep_dialog/data_movie/dia_act_nl_pairs.v7.json \ <br />
	--dqn_hidden_size 80 \ <br />
	--experience_replay_pool_size 10000 \ <br />
	--episodes 500 \ <br />
	--simulation_epoch_size 100 \ <br />
	--write_model_dir $save_path \ <br />
	--run_mode 3 \ <br />
	--act_level 0 \ <br />
	--slot_err_prob 0.0 \ <br />
	--intent_err_prob 0.0 \ <br />
	--batch_size 16 \ <br />
	--warm_start 1 \ <br />
	--warm_start_epochs 120 \ <br />
	--epsilon 0.00 \ <br />
	--gamma 0.95 \ <br />
	--dueling_dqn 0 \ <br />
	--double_dqn 0 \ <br />
	--distributional 0 \ <br />
	--icm 0 \ <br />
	--per 0 \ <br />
	--noisy 0 <br />
done  <br />

For different domains, the bash files are all stored in the _`./script`_ directory. You’ll need to modify the parameters and methods in the run files according to your needs—like _`simulation_epoch_store`_ for DDR and _`simulation_epoch_size`_ for DQN—as well as the methods called from the dialog manager, such as _`next_turn_store`_ for DDR and _`next_turn`_ for DQN."  <br />


## Reference


@article{zhao2024acl,  <br />
  title={Rescue Conversations from Dead-ends: Efficient Exploration for Task-oriented Dialogue Policy Optimization},  <br />
  author={Zhao, Yangyang and Dastani, Mehdi, and Jinchuan, Long, and Wang, Zhenyu  and Wang, Shihan},  <br />
  journal={Transactions of the Association for Computational Linguistics},  <br />
  year={2024}  <br />
}  <br />

