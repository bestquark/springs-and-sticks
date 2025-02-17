from contextlib import contextmanager

try:
    from IPython.display import display, clear_output
    IN_IPYTHON = True
except ImportError:
    IN_IPYTHON = False

@contextmanager
def verbose_display(message, verbose=True):
    if verbose:
        if IN_IPYTHON:
            # Create a display handle with a unique ID
            display(message, display_id=True)
        else:
            print(message, end="\r", flush=True)
    try:
        yield
    finally:
        if verbose:
            if IN_IPYTHON:
                clear_output(wait=True)
