import subprocess
from configparser import ConfigParser
import os
import sys

import boto3

def get_credentials():
    # add option to pass profile name
    try:
        config = ConfigParser()
        config.read(os.getenv("HOME") + "/.aws/credentials")
        return (
            config.get("default", "aws_access_key_id"),
            config.get("default", "aws_secret_access_key"),
        )
    except:
        ACCESS = os.getenv("AWS_ACCESS_KEY_ID")
        SECRET = os.getenv("AWS_SECRET_ACCESS_KEY")
    if not ACCESS and SECRET:
        raise AttributeError("No AWS credentials found.")
    return (ACCESS, SECRET)

def s3_client(service="s3"):
    """
    create an s3 client.
    Parameters
    ----------
    service : str
        Type of service.
    
    Returns
    -------
    boto3.client
        client with proper credentials.
    """

    try:
        ACCESS, SECRET = get_credentials()
    except AttributeError:
        return boto3.client(service)
    return boto3.client(service, aws_access_key_id=ACCESS, aws_secret_access_key=SECRET)


def get_matching_s3_objects(bucket, prefix="", suffix=""):
    """
    Generate objects in an S3 bucket.
    
    Parameters
    ----------
    bucket : str
        Name of the s3 bucket.
    prefix : str, optional
        Only fetch objects whose key starts with this prefix, by default ''
    suffix : str, optional
        Only fetch objects whose keys end with this suffix, by default ''
    """
    s3 = s3_client(service="s3")
    kwargs = {"Bucket": bucket}

    # If the prefix is a single string (not a tuple of strings), we can
    # do the filtering directly in the S3 API.
    if isinstance(prefix, str):
        kwargs["Prefix"] = prefix

    while True:

        # The S3 API response is a large blob of metadata.
        # 'Contents' contains information about the listed objects.
        resp = s3.list_objects_v2(**kwargs)

        try:
            contents = resp["Contents"]
        except KeyError:
            print("No contents found.")
            return

        for obj in contents:
            key = obj["Key"]
            if key.startswith(prefix) and key.endswith(suffix):
                yield key

        # The S3 API is paginated, returning up to 1000 keys at a time.
        # Pass the continuation token into the next response, until we
        # reach the final page (when this field is missing).
        try:
            kwargs["ContinuationToken"] = resp["NextContinuationToken"]
        except KeyError:
            break


def s3_get__relevant_data(bucket, remote, local, info="", file_types = ('.csv','.png') ,force=False):
    """Given and s3 directory, copies files/subdirectories in that directory to local
    Parameters
    ----------
    bucket : str
        s3 bucket you are accessing data from
    remote : str
        The path to the data on your S3 bucket. The data will be
        downloaded to the provided bids_dir on your machine.
    local : list
        Local input directory where you want the files copied to and subject/session info [input, sub-#/ses-#]
    info : str, optional
        Relevant subject and session information in the form of sub-#/ses-#
    file_types : tuple of str's, optional
        extensions of relevant file types
    force : bool, optional
        Whether to overwrite the local directory containing the s3 files if it already exists, by default False
    """
    if info == "sub-":
        print("Subject not specified, comparing input folder to remote directory...")
    else:
        if os.path.exists(os.path.join(local, info)) and not force:
            if os.listdir(os.path.join(local, info)):
                print(
                    f"Local directory: {os.path.join(local,info)} already exists. Not pulling s3 data. Delete contents to re-download data."
                )
                return  # TODO: make sure this doesn't append None a bunch of times to a list in a loop on this function
    # get client with credentials if they exist
    client = s3_client(service="s3")
    # check that bucket exists
    bkts = [bk["Name"] for bk in client.list_buckets()["Buckets"]]
    if bucket not in bkts:
        raise ValueError(
            "Error: could not locate bucket. Available buckets: " + ", ".join(bkts)
        )
    info = info.rstrip("/") + "/"
#     bpath = get_matching_s3_objects(bucket, f"{remote}/{info}")
    #test by grabbing only .csv
    #change to build larger bpath with all matching objects
    bpath = get_matching_s3_objects(bucket, suffix = file_types[1])
    # go through all folders inside of remote directory and download relevant files
    for obj in bpath:
        #check if it matches designed remote file path
        if remote in obj: 
            bdir, data = os.path.split(obj)
            localpath = os.path.join(local, bdir.replace(f"{remote}/", ""))
            #check if the file is correct type 
            if obj.endswith(file_types):
                # Make directory for data if it doesn't exist
                if not os.path.exists(localpath):
                    os.makedirs(localpath)
                if not os.path.exists(f"{localpath}/{data}"):
                    print(f"Downloading {bdir}/{data} from {bucket} s3 bucket...")
                    # Download file
                    client.download_file(bucket, f"{bdir}/{data}", f"{localpath}/{data}")
                    if os.path.exists(f"{localpath}/{data}"):
                        print("Success!")
                    else:
                        print("Error: File not downloaded")
                else:
                    print(f"File {data} already exists at {localpath}/{data}")


# client = s3_client(service="s3")

# data_dir = '/mnt/d/Downloads/neurodatadesign/output_data/s3/'


# bpath = get_matching_s3_objects(bucket = 'ndmg-data', suffix = '.csv') 
# for obj in bpath:
#         bdir, data = os.path.split(obj)
#         print(bdir)
#         print(data)

#set bucket name
bucket = 'ndmg-data'

#set local data_dir you want to dl outputs to here
data_dir = '/mnt/d/Downloads/neurodatadesign/output_data/s3/'

#set folders by filling with all folders. 
#may want to specify which run instead of downloading all runs!
# remotes = ['BNU1/', 'BNU3/']
# remotes = ['BNU1/BNU1-2-8-20-m2g_staging-native-csa-det/', 
#            'HNU1/HNU1-2-8-20-m2g_staging-native-csa-det/',
#            'NKI1/NKI1-2-8-20-m2g_staging-native-csa-det/',
#            'NKI24/NKI24-2-8-20-m2g_staging-native-csa-det/',
#            'SWU4/SWU4-2-8-20-m2g_staging-native-csa-det/']

remotes = ['HNU1/HNU1-2-8-20-m2g_staging-native-csa-det/']

#set what files to download. Current .csv (adjacency matrix) and .png (connectome graph)
file_extensions = ('.png','.csv')

for remote in remotes:
    print(remote)
    s3_get__relevant_data(bucket = 'ndmg-data', remote = remote, local = data_dir, info="", file_types = file_extensions ,force=False)
