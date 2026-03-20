import logging

def change_filehandler_mode_to_write(root_logger: logging.Logger):
    """
    Change the mode of the first FileHandler in root_logger to 'w'.
    Preserves the filename and formatter.
    """
    for h in root_logger.handlers:
        if isinstance(h, logging.FileHandler):
            fn = h.baseFilename
            fmt = h.formatter

            root_logger.removeHandler(h)
            h.close()

            new_handler = logging.FileHandler(fn, mode='w')
            new_handler.setFormatter(fmt)
            root_logger.addHandler(new_handler)
            break