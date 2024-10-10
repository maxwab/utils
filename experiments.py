import contextlib
import logging
import pickle
from pathlib import Path
from dataclasses import dataclass
import tempfile
from git import Repo
import typing as tp


@dataclass
class Experiment:
    name: str
    path: Path  # path to the root folder of the experiment


@contextlib.contextmanager
def persistent_experiment(exp_name: str, args: tp.Dict):
    """
    Creates a proper experiment, saving it in the experiments folder to be able to access results later.
    Creates a data folder where the artifacts are saved.
    Creates a code folder where the code used for the computation is copied.
    Pre-existing model checkpoints and datasets should be symlinked, for instance with `models` target name.
    @param args:
    @param exp_name:
    @return:
    """
    assert len(exp_name)  # persistent experiments should be named appropriately

    dev_folder = Path.home() / 'dev/project/'
    repo = Repo(dev_folder)
    assert not repo.is_dirty()  # ensure that all changes to git repo have been committed

    # Path management
    experiments_folder = dev_folder / 'experiments'  # experiments is a symlink to scratch!
    exp_folder = experiments_folder / exp_name
    data = exp_folder / 'data'
    if data.exists():
        print(f"{data} already exists, risk of overwriting existing data! Proceed anyway?")
        input()  # Ask for user confirmation
    data.mkdir(exist_ok=True, parents=True)  # OK to relaunch a computation within an existing folder (but there will be risk of overwriting!)
    code_folder = exp_folder / 'code'
    code_folder.mkdir(exist_ok=True, parents=True)
    with open(data / 'args.pkl', 'wb') as fd:
        pickle.dump(args, fd)

    # Set up experiment logger
    logger = logging.getLogger(exp_name)
    fh = logging.FileHandler(data / f'{exp_name}.log')  # create file handler which logs even debug messages
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)  # create console handler with a higher log level
    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)

    logger.info(f"Created a new persistent experiment at path {exp_folder}.")

    exp = Experiment(name=exp_name, path=exp_folder)
    yield exp

    logging.info("Cleanly exited.")


@contextlib.contextmanager
def temporary_experiment(exp_name: str = 'temporary_experiment', args: tp.Dict = None):
    """
    Creates a temporary experiments. The artifacts are saved in a temporary folder that will be removed after the
    computations.
    Pre-existing model checkpoints and datasets should be symlinked, for instance with `models` target name.
    @param exp_name:
    @return:
    """
    exp_folder_name = tempfile.mkdtemp()

    exp_folder = Path(exp_folder_name)  # Maybe to be created in /tmp?
    data = exp_folder / 'data'
    if data.exists():
        print(f"{data} already exists, risk of overwriting...")  # should also be logged to console
    data.mkdir(exist_ok=True,
               parents=True)  # OK to relaunch a computation within an existing folder (but there will be risk of overwriting!)
    code_folder = exp_folder / 'code'
    code_folder.mkdir(exist_ok=True, parents=True)
    with open(data / 'args.pkl', 'wb') as fd:
        pickle.dump(args, fd)

    # Set up experiment logger
    logging.basicConfig(encoding='utf-8', level=logging.INFO, handlers=[])
    logger = logging.getLogger(exp_name)
    # create file handler which logs even debug messages
    fh = logging.FileHandler(data / f'{exp_name}.log')
    fh.setLevel(logging.DEBUG)
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)

    logger.warning("This is a temporary experiment. Results will *not* be saved to disk. Proceed?")
    input()

    logger.info(f"Created a new temporary experiment at path {exp_folder}.")

    exp = Experiment(name=exp_name, path=exp_folder)
    yield exp

    # exp_folder.cleanup()  # don't cleanup to inspect the artifacts
    logger.info("Cleanly exited")
