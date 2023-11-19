import time


def timeit(message):
    """
    Decorator used to time functions and print a message
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            print(f"Starting process: {message}")
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"Process finished in: {elapsed_time:.2f} seconds")
            return result

        return wrapper

    return decorator
