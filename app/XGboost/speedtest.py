import cProfile
import pstats

from XGboost.test_recommender import test_recommender

if __name__ == "__main__":
    print("Starting profiler...")
    profiler = cProfile.Profile()
    profiler.enable()

    print("Running test_recommender...")
    test_recommender()

    profiler.disable()
    print("Profiling complete. Generating stats...")
    stats = pstats.Stats(profiler).sort_stats('cumtime')
    stats.print_stats(10)  # Print top 10 slowest functions
    print("Stats printed.")