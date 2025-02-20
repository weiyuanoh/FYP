import Solver2 as sl
import cProfile
import numpy as np

profiler = cProfile.Profile()
profiler.enable()  # Start profiling

sl.refinement_loop(0.01, np.array([0.65, 0.1]))

profiler.disable()  # Stop profiling
profiler.print_stats(sort="cumtime")  # Print the results, sorted by cumulative time
