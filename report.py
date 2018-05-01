import tensorflow as tf

# function to build summary in tensorboard
def build_summaries():
    episode_reward = tf.Variable(0.)
    tf.summary.scalar("Reward", episode_reward)
    episode_ave_max_q = tf.Variable(0.)
    tf.summary.scalar("Qmax Value", episode_ave_max_q)

    summary_vars = [episode_reward, episode_ave_max_q]
    summary_ops = tf.summary.merge_all()
    return summary_ops, summary_vars

# function to build test summary in tensorboard
def build_test_summaries():
    episode_reward = tf.Variable(0.)
    tf.summary.scalar("Reward", episode_reward)

    summary_vars = [episode_reward]
    summary_ops = tf.summary.merge_all()
    return summary_ops, summary_vars
