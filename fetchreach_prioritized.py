import os, sys, time
import json
import gym
import argparse
import numpy as np
import tensorflow as tf
from datetime import datetime
from agent import Actor, Critic
from replay import Memory
from noise import Noise
from report import build_summaries, build_test_summaries

# function to unpack observation from gym environment
def unpackObs(obs):
    return  obs['achieved_goal'], \
            obs['desired_goal'],\
            np.concatenate((obs['observation'], \
            obs['desired_goal'])), \
            np.concatenate((obs['observation'], \
            obs['achieved_goal']))

# function to train agents
def train(sess, env, args, actor, critic, actor_noise, desired_goal_dim, achieved_goal_dim, observation_dim):

    # Set path to save results
    tensorboard_dir = './' + args['env'] + '_' + args['variation'] + '/train_' + datetime.now().strftime('%Y-%m-%d-%H')
    model_dir = './' + args['env'] + '_' + args['variation'] + '/model'

    # add summary to tensorboard
    summary_ops, summary_vars = build_summaries()

    # initialize variables, create writer and saver
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    writer = tf.summary.FileWriter(tensorboard_dir, sess.graph)

    # restore session if exists
    try:
        saver.restore(sess, os.path.join(model_dir, args['env'] + '_' + args['variation'] + '.ckpt'))
        print('Restore from previous training session')
    except:
        print('Start new training session')

    # initialize target network weights and replay memory (use two memory)
    actor.update()
    critic.update()
    replay_memory = Memory(int(args['memory_size']), int(args['seed'])) # use when TD is not big
    sum_priority = 0 # sum of priority

    # train in loop
    for i in range(int(args['episodes'])):

        # reset gym, get achieved_goal, desired_goal, state
        achieved_goal, desired_goal, s, s_prime = unpackObs(env.reset())
        episode_reward = 0
        episode_maximum_q = 0

        for j in range(int(args['episode_length'])):

            # render episode
            if args['render']:
                env.render()

            # predict action and add noise
            a = actor.predict(np.reshape(s, (1, actor.state_dim)))
            a = a + actor_noise.get_noise()

            # play
            obs_next, reward, done, info = env.step(a[0])
            achieved_goal, desired_goal, state_next, state_prime_next = unpackObs(obs_next)

            # compute temporal difference error using critic and target critic network
            q0_ = critic.predict(np.reshape(s, (1, actor.state_dim)), a) # get current q
            a1_ = actor.predict(np.reshape(state_next, (1, actor.state_dim))) # predict next action for the target critic
            q1_ = critic.predict_target(np.reshape(state_next, (1, actor.state_dim)), a1_) # get target q
            td_ = np.abs(q1_[0][0] - q0_[0][0]) # compute td error
            
            # compute priority and add to sum
            priority = (td_ + args['epsilon']) ** args['alpha'] # compute priority for PER
            sum_priority = sum_priority + priority # add to sum
            p_priority = priority / sum_priority # compute probability

            # add normal experience to memory -- i.e. experience w.r.t. desired goal
            replay_memory.addWithPriority(np.reshape(s, (actor.state_dim,)), \
                        np.reshape(a, (actor.action_dim,)), \
                        reward, \
                        done, \
                        np.reshape(state_next, (actor.state_dim,)), \
                        p_priority)

            # add hindsight experience to memory -- i.e. experience w.r.t achieved goal
            substitute_goal = achieved_goal.copy()
            substitute_reward = env.compute_reward(achieved_goal, substitute_goal, info)
            replay_memory.addWithPriority(np.reshape(s_prime, (actor.state_dim,)), \
                        np.reshape(a, (actor.action_dim,)), \
                        substitute_reward, \
                        True, \
                        np.reshape(state_prime_next, (actor.state_dim,)), \
                        p_priority)

            # start to train when there's enough experience
            if replay_memory.size() > int(args['batch_size']):

                # merge batch from both buckets
                s_batch, a_batch, r_batch, d_batch, s2_batch = replay_memory.sample_batch_priority(int(args['batch_size']))
                
                # find TD -- temporal difference
                # actor find target action
                a2_batch = actor.predict_target(s2_batch)

                # critic find target q
                q2_batch = critic.predict_target(s2_batch, a2_batch)

                # add a decay of q to reward if not done
                r_batch_discounted = []
                for k in range(int(args['batch_size'])):
                    if d_batch[k]:
                        r_batch_discounted.append(r_batch[k])
                    else:
                        r_batch_discounted.append(r_batch[k] + critic.gamma * q2_batch[k])

                # train critic with state, action, and reward
                pred_q, _ = critic.train(s_batch,
                                         a_batch,
                                         np.reshape(r_batch_discounted, (int(args['batch_size']), 1)))

                # record maximum q
                episode_maximum_q += np.amax(pred_q)

                # actor find action
                a_outs = actor.predict(s_batch)

                # get comment from critic
                comment_gradients = critic.get_comment_gradients(s_batch, a_outs)

                # train actor with state and the comment gradients
                actor.train(s_batch, comment_gradients[0])

                # Update target networks
                actor.update()
                critic.update()

            # record reward and move to next state
            episode_reward += reward
            s = state_next

            # if episode ends
            if done:

                # write summary to tensorboard
                summary_str = sess.run(summary_ops, feed_dict={
                    summary_vars[0]: episode_reward,
                    summary_vars[1]: episode_maximum_q / float(j)
                })
                writer.add_summary(summary_str, i)
                writer.flush()

                # print out results
                print('| Episode: {:d} | Reward: {:d} | Q: {:.4f}'.format(i, int(episode_reward),
                                                                          (episode_maximum_q / float(j))))
                # save model
                saver.save(sess, os.path.join(model_dir, args['env'] + '_' + args['variation'] + '.ckpt'))

                break
    return

# function to test agents
def test(sess, env, args, actor, critic, desired_goal_dim, achieved_goal_dim, observation_dim):

    # Set path to save results
    tensorboard_dir = './' + args['env'] + '_' + args['variation'] + '/test_' + datetime.now().strftime('%Y-%m-%d-%H')
    model_dir = './' + args['env'] + '_' + args['variation'] + '/model'

    # add summary to tensorboard
    summary_ops, summary_vars = build_test_summaries()

    # initialize variables, create writer and saver
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    writer = tf.summary.FileWriter(tensorboard_dir, sess.graph)

    # restore session 
    try:
        saver.restore(sess, os.path.join(model_dir, args['env'] + '_' + args['variation'] + '.ckpt'))
        print('Model is trained and ready')
    except:
        print('No model. Please train first. Exit the program')
        sys.exit()

    # test in loop
    for i in range(int(args['episodes'])):

        # reset gym, get achieved_goal, desired_goal, state
        achieved_goal, desired_goal, s, s_prime = unpackObs(env.reset())
        episode_reward = 0

        for j in range(int(args['episode_length'])):

            # render episode
            if args['render']:
                env.render()

            # predict action 
            a = actor.predict(np.reshape(s, (1, actor.state_dim)))

            # play
            obs_next, reward, done, info = env.step(a[0])
            achieved_goal, desired_goal, state_next, state_prime_next = unpackObs(obs_next)

            # record reward and move to next state
            episode_reward += reward
            s = state_next

            # if episode ends
            if done:

                # write summary to tensorboard
                summary_str = sess.run(summary_ops, feed_dict={
                    summary_vars[0]: episode_reward
                })
                writer.add_summary(summary_str, i)
                writer.flush()

                # print out results
                print('| Episode: {:d} | Reward: {:d}'.format(i, int(episode_reward))) 
                
                break
    return

# Main
def main(args):

    # Set path to save result
    gym_dir = './' + args['env'] + '_' + args['variation'] + '/gym'

    # Set random seed for reproducibility
    np.random.seed(int(args['seed']))
    tf.set_random_seed(int(args['seed']))

    with tf.Session() as sess:

        # Load environment
        env = gym.make(args['env'])
        env.seed(int(args['seed']))

        # get size of action and state (i.e. output and input for the agent)
        obs = env.reset()
        observation_dim = obs['observation'].shape[0]
        achieved_goal_dim = obs['achieved_goal'].shape[0]
        desired_goal_dim =  obs['desired_goal'].shape[0]
        assert achieved_goal_dim == desired_goal_dim

        # state size = observation size + goal size
        state_dim = observation_dim + desired_goal_dim
        action_dim = env.action_space.shape[0]
        action_highbound = env.action_space.high

        # print out parameters
        print('Parameters:')
        print('Observation Size=', observation_dim)
        print('Goal Size=', desired_goal_dim)
        print('State Size =', state_dim)
        print('Action Size =', action_dim)
        print('Action Upper Boundary =', action_highbound)

        # save to monitor if render
        if args['render']:
            env = gym.wrappers.Monitor(env, gym_dir, force=True)
        else:
            env = gym.wrappers.Monitor(env, gym_dir, video_callable=False, force=True)

        # create actor
        actor = Actor(sess, state_dim, action_dim, action_highbound,
                      float(args['actor_lr']), float(args['tau']),
                      int(args['batch_size']), int(args['hidden_size']))

        # create critic
        critic = Critic(sess, state_dim, action_dim,
                        float(args['critic_lr']), float(args['tau']),
                        float(args['gamma']),
                        actor.n_actor_vars,
                        int(args['hidden_size']))

        # noise
        actor_noise = Noise(mu=np.zeros(action_dim))

        # train the network
        if not args['test']:
            train(sess, env, args, actor, critic, actor_noise, desired_goal_dim, achieved_goal_dim, observation_dim)
        else:
            test(sess, env, args, actor, critic, desired_goal_dim, achieved_goal_dim, observation_dim)

        # close gym
        env.close()

        # close session
        sess.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # training parameters
    parser.add_argument('--actor-lr', help='actor learning rate', default=0.0001)
    parser.add_argument('--critic-lr', help='critic learning rate', default=0.001)
    parser.add_argument('--batch-size', help='batch size', default=64)
    parser.add_argument('--gamma', help='discount factor reward', default=0.99)
    parser.add_argument('--tau', help='target update tau', default=0.001)
    parser.add_argument('--memory-size', help='size of the replay memory', default=1000000)
    parser.add_argument('--hidden-size', help='number of nodes in hidden layer', default=256)
    parser.add_argument('--episodes', help='episodes to train', default=2000)
    parser.add_argument('--episode-length', help='max length of 1 episode', default=1000)
    parser.add_argument('--epsilon', help='constant for td error -- any small number', default=0.1)
    parser.add_argument('--alpha', help='priority factor range from zero to one -- zero means uniform', default=0.5)
    
    # others and defaults
    parser.add_argument('--seed', help='random seed', default=1234)
    parser.add_argument('--render', help='render the gym env', action='store_true')
    parser.add_argument('--test', help='test mode does not do exploration', action='store_true')
    parser.add_argument('--variation', help='model variation name', default='DDPG_HER_PER')
    parser.set_defaults(env='FetchReach-v1')
    parser.set_defaults(render=False)
    parser.set_defaults(test=False)

    # parse arguments
    args = vars(parser.parse_args())

    # run main
    start_time = time.time()
    main(args)
    print("--- %s seconds ---" % (time.time() - start_time))
