import datetime
import logging
import os


def get_log(log_dir, annotation):
    # log part https://www.jb51.net/article/88449.htm
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    logger = logging.getLogger(annotation)
    logger.setLevel(logging.INFO)
    dd = datetime.datetime.now()
#    fh = logging.FileHandler(log_dir + "out_project_%s.log.%s" % (annotation, dd.isoformat()))
    fh = logging.FileHandler(os.path.join(log_dir,"out_project_%s_%s.log" % (annotation, dd.strftime('%Y-%m-%d-%H_%M_%S'))))
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    # log format
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)
    logger.addHandler(ch)
    logger.addHandler(fh)
    # log part end
    return logger
