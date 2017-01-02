from markovchain import *
from training_algorithms import *

# Builds a formatted string from the generator
def gen(m):
    return ''.join([w for w in m.generate_formatted(word_wrap=60, soft_wrap=True, start_with=None, max_len=100, verbose=True)])

# Initialize the chain and train it on a few of my reddit posts.
mkv = MarkovChain()
mkv.bulk_train('training_data\\*.txt', verbose=True)
# Or,
# mkv.load_training('training01.txt')

# Store this information for later, so that there's no need to re-train the next time.
mkv.save_training('stored_data\\training')
# Adjust the weights with the help of some fitness functions.
mkv.bulk_adjust_weights(fitness_functions=[aw_favor_alternating_complexity, aw_mul(aw_favor_punctuation, 0.5)],
                        iterations=10000,
                        verbose=True)
# Save the new state to a different file, to prevent feedback loops.
mkv.save_training('stored_data\\training01')
# Print a sample output after all weights have been adjusted.
print(gen(mkv))