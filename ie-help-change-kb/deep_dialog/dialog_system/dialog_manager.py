"""
Created on May 17, 2016

@author: xiul, t-zalipt
"""

import json, copy
from . import StateTracker
from deep_dialog import dialog_config
import math
from deep_dialog.dialog_system import StateIndexStore
class DialogManager:
    """ A dialog manager to mediate the interaction between an agent and a customer """
    
    def __init__(self, agent, user, act_set, slot_set, movie_dictionary):
        self.agent = agent
        self.user = user
        self.act_set = act_set
        self.slot_set = slot_set
        self.state_tracker = StateTracker(act_set, slot_set, movie_dictionary)
        self.user_action = None
        self.reward = 0
        self.instrinsic_reward = 0
        self.episode_over = False
        self.state_index_store = StateIndexStore()
    def initialize_episode_store(self, selected_goal=None):
        """ Refresh state for new dialog """

        self.reward = 0
        self.instrinsic_reward = 0
        self.episode_over = False
        self.state_tracker.initialize_episode()
        self.user_action = self.user.initialize_episode(selected_goal=selected_goal)
        self.state_tracker.update(user_action=self.user_action)

        if dialog_config.run_mode < 3:
            print ("New episode, user goal:")
            print json.dumps(self.user.goal, indent=2)
        self.print_function(user_action=self.user_action)

        self.agent.initialize_episode()

        return self.user_action['nl']

    def initialize_episode(self, selected_goal=None):
        """ Refresh state for new dialog """
        
        self.reward = 0
        self.instrinsic_reward = 0
        self.episode_over = False
        self.state_tracker.initialize_episode()
        self.user_action = self.user.initialize_episode(selected_goal=selected_goal)
        self.state_tracker.update(user_action = self.user_action)
        
        if dialog_config.run_mode < 3:
            print ("New episode, user goal:")
            print json.dumps(self.user.goal, indent=2)
        self.print_function(user_action = self.user_action)
            
        self.agent.initialize_episode()

    def next_turn(self, record_training_data=True):
        """ This function initiates each subsequent exchange between agent and user (agent first) """
        if self.agent.warm_start == 1:
            # n_old = len(self.state_tracker.get_current_kb_results())
            ########################################################################
            #   CALL AGENT TO TAKE HER TURN
            ########################################################################
            self.state = self.state_tracker.get_state_for_agent()
            self.agent_action = self.agent.state_to_action(self.state)
            ########################################################################
            #   Register AGENT action with the state_tracker
            ########################################################################
            self.state_tracker.update(agent_action=self.agent_action)
            self.agent.add_nl_to_action(self.agent_action) # add NL to Agent Dia_Act
            self.print_function(agent_action = self.agent_action['act_slot_response'])
            ########################################################################
            #   CALL USER TO TAKE HER TURN
            ########################################################################
            self.sys_action = self.state_tracker.dialog_history_dictionaries()[-1]
            self.user_action, self.episode_over, dialog_status = self.user.next(self.sys_action)
            self.reward = self.reward_function(dialog_status)
            ########################################################################
            #   Update state tracker with latest user action
            ########################################################################
            if self.episode_over != True:
                self.state_tracker.update(user_action = self.user_action)
                self.print_function(user_action = self.user_action)

                # n_new = len(self.state_tracker.get_current_kb_results())
                # if n_new<n_old and n_new>0 :
                #     self.reward += math.log(n_old,2)-math.log(n_new,2)
                # if n_new==0 and n_old!=0 :
                #     self.reward = -self.user.max_turn
            ########################################################################
            #  Inform agent of the outcome for this timestep (s_t, a_t, r, s_{t+1}, episode_over)
            ########################################################################
            if record_training_data:
                self.agent.register_experience_replay_tuple(self.state, self.agent_action, self.reward, self.state_tracker.get_state_for_agent(), self.episode_over)
        else:
            n_old = len(self.state_tracker.get_current_kb_results())
            ########################################################################
            #   CALL AGENT TO TAKE HER TURN
            ########################################################################
            self.state = self.state_tracker.get_state_for_agent()
            self.agent_action = self.agent.state_to_action(self.state)

            ########################################################################
            #   Register AGENT action with the state_tracker
            ########################################################################
            self.state_tracker.update(agent_action=self.agent_action)

            self.agent.add_nl_to_action(self.agent_action) # add NL to Agent Dia_Act
            self.print_function(agent_action = self.agent_action['act_slot_response'])

            ########################################################################
            #   CALL USER TO TAKE HER TURN
            ########################################################################
            self.sys_action = self.state_tracker.dialog_history_dictionaries()[-1]
            self.user_action, self.episode_over, dialog_status = self.user.next(self.sys_action)
            self.reward = self.reward_function(dialog_status)
            #self.instrinsic_reward = self.agent.get_intrinsic_reward(self.state, self.state_tracker.get_state_for_agent(), self.agent_action)

            ########################################################################
            #   Update state tracker with latest user action
            ########################################################################
            if self.episode_over != True:
                self.state_tracker.update(user_action = self.user_action)
                self.print_function(user_action = self.user_action)

                self.state_tracker.store()
                self.user.store()

                n_new = len(self.state_tracker.get_current_kb_results())
                if n_new>n_old:
                    print "increase:",n_old,"to",n_new

                if n_new==0 and n_old!=0 :
                    self.reward = -self.user.max_turn
                    if record_training_data:
                        self.agent.register_experience_replay_tuple(self.state, self.agent_action, self.reward,
                                                                    self.state_tracker.get_state_for_agent(),
                                                                    self.episode_over)
                    dead_end_index = len(self.agent.experience_replay_pool)-1
                    self.state_index_store.add(state, dead_end_index)
                    min_N = float('inf')
                    min_N_index = -1
                    for i in range(self.agent.num_actions):
                        self.state_tracker.back_state()
                        self.user.back()
                        self.agent_action = self.agent.to_action(i)
                        self.state_tracker.update(agent_action=self.agent_action)
                        self.agent.add_nl_to_action(self.agent_action)
                        self.print_function(agent_action=self.agent_action['act_slot_response'])
                        self.sys_action = self.state_tracker.dialog_history_dictionaries()[-1]
                        self.user_action, self.episode_over, dialog_status = self.user.next(self.sys_action)
                        self.reward = self.reward_function(dialog_status)
                        n = len(self.state_tracker.get_current_kb_results())
                        # H_n = -n_old*(1/n_old)*math.log(n_old, 2)-((-n_new/n_old)*1/n_new)*math.log(n_new,2)
                        if n!=0 and n<min_N:
                        # if H_n!=0 and n<min_N:
                        #     min_N=H_n
                            min_N=n
                            min_N_index=i
                    if min_N_index!=-1:
                        self.state_tracker.back_state()
                        self.user.back()
                        self.agent_action = self.agent.to_action(min_N_index)
                        self.state_tracker.update(agent_action=self.agent_action)
                        self.agent.add_nl_to_action(self.agent_action)
                        self.print_function(agent_action=self.agent_action['act_slot_response'])
                        self.sys_action = self.state_tracker.dialog_history_dictionaries()[-1]
                        self.user_action, self.episode_over, dialog_status = self.user.next(self.sys_action)
                        self.reward = self.reward_function(dialog_status)
                        # n_new = len(self.state_tracker.get_current_kb_results())
                        # if n_new < n_old and n_new > 0:
                        #     self.reward += math.log(n_old, 2) - math.log(n_new, 2)
                        if record_training_data:
                            self.agent.register_experience_replay_tuple(self.state, self.agent_action, self.reward,
                                                                        self.state_tracker.get_state_for_agent(),
                                                                        self.episode_over)
                    else:
                        self.state_tracker.recover()
                        self.user.recover()
                else:
                    # if n_new < n_old and n_new>0 :
                    #     self.reward += math.log(n_old,2)-math.log(n_new,2)
                    if record_training_data:
                        self.agent.register_experience_replay_tuple(self.state, self.agent_action, self.reward,
                                                                    self.state_tracker.get_state_for_agent(),
                                                                    self.episode_over)
            else:
                if record_training_data:
                    self.agent.register_experience_replay_tuple(self.state, self.agent_action, self.reward, self.state_tracker.get_state_for_agent(), self.episode_over)
        
        return (self.episode_over, self.reward)

    def next_turn_store(self, record_training_data=True):
        """ This function initiates each subsequent exchange between agent and user (agent first) """
        if self.agent.warm_start == 1:
            # n_old = len(self.state_tracker.get_current_kb_results())
            ########################################################################
            #   CALL AGENT TO TAKE HER TURN
            ########################################################################
            self.state = self.state_tracker.get_state_for_agent()
            self.agent_action = self.agent.state_to_action(self.state)
            ########################################################################
            #   Register AGENT action with the state_tracker
            ########################################################################
            self.state_tracker.update(agent_action=self.agent_action)
            self.agent.add_nl_to_action(self.agent_action)  # add NL to Agent Dia_Act
            self.print_function(agent_action=self.agent_action['act_slot_response'])
            ########################################################################
            #   CALL USER TO TAKE HER TURN
            ########################################################################
            self.sys_action = self.state_tracker.dialog_history_dictionaries()[-1]
            self.user_action, self.episode_over, dialog_status = self.user.next(self.sys_action)
            self.reward = self.reward_function(dialog_status)
            ########################################################################
            #   Update state tracker with latest user action
            ########################################################################
            if self.episode_over != True:
                self.state_tracker.update(user_action=self.user_action)
                self.print_function(user_action=self.user_action)

                # n_new = len(self.state_tracker.get_current_kb_results())
                # if n_new<n_old and n_new>0 :
                #     self.reward += math.log(n_old,2)-math.log(n_new,2)
                # if n_new==0 and n_old!=0 :
                #     self.reward = -self.user.max_turn
            ########################################################################
            #  Inform agent of the outcome for this timestep (s_t, a_t, r, s_{t+1}, episode_over)
            ########################################################################
            if record_training_data:
                self.agent.register_experience_replay_tuple(self.state, self.agent_action, self.reward,
                                                            self.state_tracker.get_state_for_agent(), self.episode_over)
        else:
            n_old = len(self.state_tracker.get_current_kb_results())
            ########################################################################
            #   CALL AGENT TO TAKE HER TURN
            ########################################################################
            self.state = self.state_tracker.get_state_for_agent()
            self.agent_action = self.agent.state_to_action(self.state)

            ########################################################################
            #   Register AGENT action with the state_tracker
            ########################################################################
            self.state_tracker.update(agent_action=self.agent_action)

            self.agent.add_nl_to_action(self.agent_action)  # add NL to Agent Dia_Act
            self.print_function(agent_action=self.agent_action['act_slot_response'])

            ########################################################################
            #   CALL USER TO TAKE HER TURN
            ########################################################################
            self.sys_action = self.state_tracker.dialog_history_dictionaries()[-1]
            self.user_action, self.episode_over, dialog_status = self.user.next(self.sys_action)
            self.reward = self.reward_function(dialog_status)
            # self.instrinsic_reward = self.agent.get_intrinsic_reward(self.state, self.state_tracker.get_state_for_agent(), self.agent_action)

            n_new_new = len(self.state_tracker.get_current_kb_results())
            ########################################################################
            #   Update state tracker with latest user action
            ########################################################################
            if self.episode_over != True:
                self.state_tracker.update(user_action=self.user_action)
                self.print_function(user_action=self.user_action)

                self.state_tracker.store()
                self.user.store()

                n_new = len(self.state_tracker.get_current_kb_results())
                if n_new > n_old:
                    print "increase:", n_old, "to", n_new

                if n_new == 0 and n_old != 0:
                    # self.reward = -self.user.max_turn
                    if record_training_data:
                        self.agent.register_experience_replay_tuple(self.state, self.agent_action, self.reward,
                                                                    self.state_tracker.get_state_for_agent(),
                                                                    self.episode_over)


                    min_N = float('inf')
                    min_N_index = -1
                    for i in range(self.agent.num_actions):
                        self.state_tracker.back_state()
                        self.user.back()
                        self.agent_action = self.agent.to_action(i)
                        self.state_tracker.update(agent_action=self.agent_action)
                        self.agent.add_nl_to_action(self.agent_action)
                        self.print_function(agent_action=self.agent_action['act_slot_response'])
                        self.sys_action = self.state_tracker.dialog_history_dictionaries()[-1]
                        self.user_action, self.episode_over, dialog_status = self.user.next(self.sys_action)
                        self.reward = self.reward_function(dialog_status)
                        n = len(self.state_tracker.get_current_kb_results())
                        if n != 0 and n < min_N:
                            min_N = n
                            min_N_index = i
                    if min_N_index != -1:
                        self.state_tracker.back_state()
                        self.user.back()
                        self.agent_action = self.agent.to_action(min_N_index)
                        self.state_tracker.update(agent_action=self.agent_action)
                        self.agent.add_nl_to_action(self.agent_action)
                        self.print_function(agent_action=self.agent_action['act_slot_response'])
                        self.sys_action = self.state_tracker.dialog_history_dictionaries()[-1]
                        self.user_action, self.episode_over, dialog_status = self.user.next(self.sys_action)
                        self.reward = self.reward_function(dialog_status)
                        # n_new = len(self.state_tracker.get_current_kb_results())
                        # if n_new < n_old and n_new > 0:
                        #     self.reward += math.log(n_old, 2) - math.log(n_new, 2)
                        if record_training_data:
                            self.agent.register_experience_replay_tuple(self.state, self.agent_action, self.reward,
                                                                        self.state_tracker.get_state_for_agent(),
                                                                        self.episode_over)
                    else:
                        self.state_tracker.recover()
                        self.user.recover()
                else:
                    # if n_new < n_old and n_new>0 :
                    #     self.reward += math.log(n_old,2)-math.log(n_new,2)
                    if record_training_data:
                        self.agent.register_experience_replay_tuple(self.state, self.agent_action, self.reward,
                                                                    self.state_tracker.get_state_for_agent(),
                                                                    self.episode_over)
            else:
                if record_training_data:
                    self.agent.register_experience_replay_tuple(self.state, self.agent_action, self.reward,
                                                                self.state_tracker.get_state_for_agent(),
                                                                self.episode_over)

        return (self.episode_over, self.reward, n_old, n_new_new, self.agent_action['act_slot_response']['nl'],
                self.user_action['nl'])

    def reward_function(self, dialog_status):
        """ Reward Function 1: a reward function based on the dialog_status """
        if dialog_status == dialog_config.FAILED_DIALOG:
            reward = -self.user.max_turn
        elif dialog_status == dialog_config.SUCCESS_DIALOG:
            reward = 2*self.user.max_turn
        else:
            reward = -1
        return reward
    
    def reward_function_without_penalty(self, dialog_status):
        """ Reward Function 2: a reward function without penalty on per turn and failure dialog """
        if dialog_status == dialog_config.FAILED_DIALOG:
            reward = 0
        elif dialog_status == dialog_config.SUCCESS_DIALOG:
            reward = 2*self.user.max_turn
        else:
            reward = 0
        return reward
    
    
    def print_function(self, agent_action=None, user_action=None):
        """ Print Function """
            
        if agent_action:
            if dialog_config.run_mode == 0:
                if self.agent.__class__.__name__ != 'AgentCmd':
                    print ("Turn %d sys: %s" % (agent_action['turn'], agent_action['nl']))
            elif dialog_config.run_mode == 1:
                if self.agent.__class__.__name__ != 'AgentCmd':
                    print ("Turn %d sys: %s, inform_slots: %s, request slots: %s" % (agent_action['turn'], agent_action['diaact'], agent_action['inform_slots'], agent_action['request_slots']))
            elif dialog_config.run_mode == 2: # debug mode
                print ("Turn %d sys: %s, inform_slots: %s, request slots: %s" % (agent_action['turn'], agent_action['diaact'], agent_action['inform_slots'], agent_action['request_slots']))
                print ("Turn %d sys: %s" % (agent_action['turn'], agent_action['nl']))
            
            if dialog_config.auto_suggest == 1:
                print('(Suggested Values: %s)' % (self.state_tracker.get_suggest_slots_values(agent_action['request_slots'])))
              
        elif user_action:
            if dialog_config.run_mode == 0:
                print ("Turn %d usr: %s" % (user_action['turn'], user_action['nl']))
            elif dialog_config.run_mode == 1: 
                print ("Turn %s usr: %s, inform_slots: %s, request_slots: %s" % (user_action['turn'], user_action['diaact'], user_action['inform_slots'], user_action['request_slots']))
            elif dialog_config.run_mode == 2: # debug mode, show both
                print ("Turn %d usr: %s, inform_slots: %s, request_slots: %s" % (user_action['turn'], user_action['diaact'], user_action['inform_slots'], user_action['request_slots']))
                print ("Turn %d usr: %s" % (user_action['turn'], user_action['nl']))
            
            if self.agent.__class__.__name__ == 'AgentCmd': # command line agent
                user_request_slots = user_action['request_slots']
                if 'ticket'in user_request_slots.keys(): del user_request_slots['ticket']
                
                if 'reservation' in user_request_slots.keys(): del user_request_slots['reservation']
                if 'taxi' in user_request_slots.keys(): del user_request_slots['taxi']
                
                if len(user_request_slots) > 0:
                    possible_values = self.state_tracker.get_suggest_slots_values(user_action['request_slots'])
                    for slot in possible_values.keys():
                        if len(possible_values[slot]) > 0:
                            print('(Suggested Values: %s: %s)' % (slot, possible_values[slot]))
                        elif len(possible_values[slot]) == 0:
                            print('(Suggested Values: there is no available %s)' % (slot))
                else:
                    pass
                  
                    #kb_results = self.state_tracker.get_current_kb_results()
                    #print ('(Number of movies in KB satisfying current constraints: %s)' % len(kb_results))

