"""
    help functions for file processing
    Copyright 2019 Yang Kaiyu https://github.com/yku12cn
"""
from pathlib import Path
import os
import pickle
from io import BytesIO
from hashlib import md5

from PIL import Image, UnidentifiedImageError

# These are cache files created by this module
_ignoreType = (".vdcache", ".vdsnap", ".vdmeta")


def inspectFileList(inset, relative=False):
    r"""Return a sorted list of file names in given directory

    Args:
        inset (str/Path): the inset path of the package
        relative (bool, optional): Return relative path or not.

    Returns:
        [list of Path]: a sorted list of file names
    """
    if not isinstance(inset, Path):
        inset = Path(inset)

    _files = []
    for file in os.scandir(inset):  # os.scandir is slightly faster
        if file.is_file() and not file.name.endswith(_ignoreType):
            if relative:
                _files.append(Path(file).relative_to(inset))
            else:
                _files.append(Path(file))

    _files.sort()
    return _files


def detectImgMode(filelist):
    r"""Auto detect image type

    Args:
        filelist (list of Path): images to be analysed

    Returns:
        str: one of PIL built in modes ('L','P','RGB'...)
    """
    for file in filelist:
        if file.is_dir():  # Jump over dir
            continue
        try:
            with Image.open(file) as testimg:
                return testimg.mode
        except UnidentifiedImageError:
            continue
    return None


def inspectFileMeta(inset):
    r"""Return a sorted list of file metas in given directory

    Args:
        inset (str/Path): the inset path of the package

    Returns:
        [list of tuple]: [(filename1, filemeta1),(filename2, filemeta2)...]
    """
    if not isinstance(inset, Path):
        inset = Path(inset)
    _files = []

    for file in os.scandir(inset):  # os.scandir is slightly faster
        if file.is_file() and not file.name.endswith(_ignoreType):
            _fstat = file.stat()
            _files.append((
                file.name,
                _fstat.st_size,
                _fstat.st_mtime_ns,
                _fstat.st_ctime_ns
            ))

    _files.sort(key=lambda x: x[0])
    return _files


def calMD5(infile, chunk_size=1024 * 1024):
    r"""calculate MD5 hash for file-like obj or variable

    Args:
        infile (file-like obj or variable): MD5 of what.
        chunk_size (int, optional): HASH chunk size. Defaults to 1024*1024.

    Returns:
        str: MD5 hexdigest
    """
    # Trun variable into file-like obj
    temp = None
    if not hasattr(infile, 'read'):
        temp = BytesIO()
        pickle.dump(infile, temp)
        infile = temp

    # Cal MD5
    f_hash = md5()
    infile.seek(0)
    for chunk in iter(lambda: infile.read(chunk_size), b''):
        f_hash.update(chunk)
    infile.seek(0)

    if temp:
        temp.close()
    return f_hash.hexdigest()
