###############################################################################
#  Benchmark script for Modin (a pandas replacement)                          #
#                                                                             #
# (c) Simon Wenkel                                                            #
# released under a 3-clause BSD license                                       #
#                                                                             #
###############################################################################


###############################################################################
#                                                                             #
#                                                                             #
# import libraries                                                            #
import time
scriptStartTime = time.time()
import sys
import os
import numpy as np
from tqdm import tqdm
import argparse
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import wget
import zipfile
import hashlib
import gc
import gzip
import shutil
import random
import pickle
#                                                                             #
#                                                                             #
###############################################################################


###############################################################################
#                                                                             #
#                                                                             #
# function and classes                                                        #

def seed_everything(seed):
    """
    Getting rid of all the randomness in the world :(
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)

def parseARGS():
    """
    Parsing args and generate config file
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-df", "--dataFolder", type=str, default="./data/",
                        help="Folder to store datasets (default=./data/)")
    parser.add_argument("-of", "--outputFolder", type=str, default="./outputs/",
                        help="Folder to store outputs (default=./outputs/)")
    parser.add_argument("-j", "--jobs", type=int, default=-1,
                        help="Number of threads to be used (default=-1)")
    parser.add_argument("-uP","--usePandas", type=bool, default=True,
                        help="Benchmark pandas (default=True)")
    parser.add_argument("-uM","--useModin", type=bool, default=True,
                        help="Benchmark Modin (default=True)")
    parser.add_argument("-mb", "--ModinBackend", type=str, default="auto",
                        help="Backend for Modin {auto,dask,ray} (default=auto)")
    config={}
    args = parser.parse_args()
    config["dataFolder"] = args.dataFolder
    config["outputFolder"] = args.outputFolder
    config["jobs"] = args.jobs # not implemented yet
    config["usePandas"] = args.usePandas
    config["useModin"] = args.useModin
    config["ModinBackend"] = args.ModinBackend
    return config

def checkGenerateFolders(config:dict):
    os.makedirs(config["dataFolder"],  exist_ok=True)
    os.makedirs(config["outputFolder"],  exist_ok=True)

def downloadDatasets(config:dict):
    """
    Downloads datasets if not existing or checksum is incorrect
    """
    def checkCHECKSUM(file):
        """
        Check if correct file was downloaded (md5sum is enough for that)
        """
        f = open(file, "rb")
        try:
            fileHash =  hashlib.md5(f.read()).hexdigest()
        finally:
            f.close()
        return fileHash

    for dataset in tqdm(CONSTANTS["Datasets"]):
        """
        Correct MD5 checksum only implemented for non-converted files
        """
        downloadStatus = False
        if os.path.isfile(config["dataFolder"]+dataset+".csv"):
            print(config["dataFolder"]+dataset+".csv exists already.")
            if checkCHECKSUM(config["dataFolder"]+dataset+".csv") != CONSTANTS["Datasets"][dataset]["MD5SUM"]:
                os.remove(config["dataFolder"]+dataset+".csv")
                print("Incorrect checksum. File broken. Re-download initiated.")
            else:
                downloadStatus = True
        if not downloadStatus:
            print("Downloading", CONSTANTS["Datasets"][dataset]["URL"])
            fileExt = "."+CONSTANTS["Datasets"][dataset]["URL"].split("/")[-1].split(".")[-1]
            wget.download(CONSTANTS["Datasets"][dataset]["URL"],
                          out=config["dataFolder"]+dataset+fileExt)
            if checkCHECKSUM(config["dataFolder"]+dataset+fileExt) != CONSTANTS["Datasets"][dataset]["MD5SUM"]:
                os.remove(config["dataFolder"]+dataset+fileExt)
                raise ValueError("Incorrect checksum of "+config["dataFolder"]+dataset+fileExt+ \
                            ". File deleted. Dataset generation aborted.")
            if (fileExt == ".xlsx") or (fileExt == ".xls"):
                # no native modin excel support so far
                print("\nConverting Excel file to csv\n")
                import pandas as pd
                tmpDF = pd.read_excel(config["dataFolder"]+dataset+fileExt)
                tmpDF.to_csv(config["dataFolder"]+dataset+".csv", index=False)
            if fileExt == ".gz":
                 with gzip.open(config["dataFolder"]+dataset+fileExt, "rb") as inputFile:
                        with open(config["dataFolder"]+dataset+".csv", "wb") as outputFile:
                            shutil.copyfileobj(inputFile, outputFile)
            if CONSTANTS["Datasets"][dataset]["Delimiter"] != "auto":
                # don't want to deal with delimiter while benchmarking
                import pandas as pd
                tmpDF = pd.read_excel(config["dataFolder"]+dataset+fileExt)
                tmpDF.to_csv(config["dataFolder"]+dataset+".csv", index=False)
            downloadStatus = True





def runBechmark(config:dict):
    """
    run bechnmarks
    """
    def benchmarks(config:dict, file:str, dataset:str, sw:str):
        """
        currently supported:
            1. load time
            2. time to calculate basic statistics (df.describe())
            3. dump time
        """
        starttime = time.time()
        df = pd.read_csv(file)
        loadTime = time.time()-starttime

        starttime = time.time()
        df_des = df.describe()
        describeTime = time.time()-starttime

        starttime = time.time()
        df.to_csv(config["outputFolder"]+dataset+".csv", index=False)
        dumpTime = time.time()-starttime
        result = [sw, dataset, loadTime, describeTime, dumpTime]
        return result

    results = {}
    resCounter = 0

    if config["usePandas"]:
        print("Using pandas")
        import pandas as pd
        print("pandas version:", pd.__version__)
        sw = "pandas"
        for dataset in tqdm(CONSTANTS["Datasets"]):
            results[resCounter] = benchmarks(config=config,
                                             file=config["dataFolder"]+dataset+".csv",
                                             dataset=dataset,
                                             sw=sw)
            resCounter += 1

    if config["useModin"]:
        print("Using Modin")
        if config["ModinBackend"] != "auto":
            os.environ["MODIN_ENGINE"] = config["ModinBackend"]
        import modin.pandas as pd
        print("Modin version:", pd.__version__)
        sw = "modin"
        for dataset in tqdm(CONSTANTS["Datasets"]):
            results[resCounter] = benchmarks(config=config,
                                             file=config["dataFolder"]+dataset+".csv",
                                             dataset=dataset,
                                             sw=sw)
            resCounter += 1
    return results

#                                                                             #
#                                                                             #
###############################################################################


###############################################################################
#                                                                             #
#                                                                             #
# CONSTANTS                                                                   #
CONSTANTS = {}
CONSTANTS["Datasets"] = {}
CONSTANTS["Datasets"]["Wholesale_customers"] = {}
CONSTANTS["Datasets"]["Wholesale_customers"]["Description"] = "https://archive.ics.uci.edu/ml/datasets/Wholesale+customers"
CONSTANTS["Datasets"]["Wholesale_customers"]["URL"] = "https://archive.ics.uci.edu/ml/machine-learning-databases/00292/Wholesale customers data.csv"
CONSTANTS["Datasets"]["Wholesale_customers"]["MD5SUM"] = "a77e467b90b10ec157fef6de32feef05"
CONSTANTS["Datasets"]["Wholesale_customers"]["Delimiter"] = "auto"
CONSTANTS["Datasets"]["Electricity_grid_stability"] = {}
CONSTANTS["Datasets"]["Electricity_grid_stability"]["Description"] = "https://archive.ics.uci.edu/ml/datasets/Electrical+Grid+Stability+Simulated+Data+"
CONSTANTS["Datasets"]["Electricity_grid_stability"]["URL"] = "https://archive.ics.uci.edu/ml/machine-learning-databases/00471/Data_for_UCI_named.csv"
CONSTANTS["Datasets"]["Electricity_grid_stability"]["MD5SUM"] = "7dbe5ce5be92325f1b2d7a08bf356f35"
CONSTANTS["Datasets"]["Electricity_grid_stability"]["Delimiter"] = "auto"
CONSTANTS["Datasets"]["OnlineRetail"] = {}
CONSTANTS["Datasets"]["OnlineRetail"]["Description"] = "https://archive.ics.uci.edu/ml/datasets/Online+Retail"
CONSTANTS["Datasets"]["OnlineRetail"]["URL"] = "https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online Retail.xlsx"
CONSTANTS["Datasets"]["OnlineRetail"]["MD5SUM"] = "8f8e6d94ba88f976f4d8290cb2dea7fd"
CONSTANTS["Datasets"]["OnlineRetail"]["Delimiter"] = "auto"
CONSTANTS["Datasets"]["OnlineRetail2"] = {}
CONSTANTS["Datasets"]["OnlineRetail2"]["Description"] = "https://archive.ics.uci.edu/ml/datasets/Online+Retail+II"
CONSTANTS["Datasets"]["OnlineRetail2"]["URL"] = "https://archive.ics.uci.edu/ml/machine-learning-databases/00502/online_retail_II.xlsx"
CONSTANTS["Datasets"]["OnlineRetail2"]["MD5SUM"] = "ed54ccfc5d358481c399cc11d0a244be"
CONSTANTS["Datasets"]["OnlineRetail2"]["Delimiter"] = "auto"
CONSTANTS["Datasets"]["WeightLifting"] = {}
CONSTANTS["Datasets"]["WeightLifting"]["Description"] = "https://archive.ics.uci.edu/ml/datasets/Weight+Lifting+Exercises+monitored+with+Inertial+Measurement+Units"
CONSTANTS["Datasets"]["WeightLifting"]["URL"] = "https://archive.ics.uci.edu/ml/machine-learning-databases/00273/Example_WearableComputing_weight_lifting_exercises_biceps_curl_variations.csv"
CONSTANTS["Datasets"]["WeightLifting"]["MD5SUM"] = "96772a813acedda8f294fd64f5d12648"
CONSTANTS["Datasets"]["WeightLifting"]["Delimiter"] = "auto"
CONSTANTS["Datasets"]["MetroInterstateTraffic"] = {}
CONSTANTS["Datasets"]["MetroInterstateTraffic"]["Description"] = "https://archive.ics.uci.edu/ml/datasets/Metro+Interstate+Traffic+Volume"
CONSTANTS["Datasets"]["MetroInterstateTraffic"]["URL"] = "https://archive.ics.uci.edu/ml/machine-learning-databases/00492/Metro_Interstate_Traffic_Volume.csv.gz"
CONSTANTS["Datasets"]["MetroInterstateTraffic"]["MD5SUM"] = "138299b75bf3b6f74cf039362e0d18f1"
CONSTANTS["Datasets"]["MetroInterstateTraffic"]["Delimiter"] = "auto"
#                                                                             #
#                                                                             #
###############################################################################



###############################################################################
#                                                                             #
#                                                                             #
# main function                                                               #
def main():
    print("=" * 80)
    config = parseARGS()
    seed_everything(1)
    if config["ModinBackend"] != "auto":
        os.environ["MODIN_ENGINE"] = config["ModinBackend"]
    checkGenerateFolders(config)
    print("-" * 80)
    print("Downloading datasets")
    startTime = time.time()
    downloadDatasets(config)
    print("Datasets downloaded in {:.2f} min.".format((time.time()-startTime)/60))
    print("-" * 80)
    startTime = time.time()
    results = runBechmark(config)
    pickle.dump(results, open(config["outputFolder"]+"results.p","wb"))
    header = ["Software", "Dataset", "loadTime", "describeTime", "dumpTime"]
    import pandas as pd
    resultsDF = pd.DataFrame.from_dict(results, orient="index", columns=header)
    resultsDF.to_csv(config["outputFolder"]+"results.csv")
    print(resultsDF)
    print("-" * 80)
    print("Total runtime: {:.2f} min".format((time.time()-scriptStartTime)/60))
    print("=" * 80)
#                                                                             #
#                                                                             #
###############################################################################

if __name__ == "__main__":
    main()
