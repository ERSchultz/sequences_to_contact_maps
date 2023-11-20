HG19_BAD_REGIONS = {1:'0-3,120-150',
                    2:'85-97',
                    3:'90-95',
                    4:'48-54',
                    5:'45-50,67-72',
                    6:'57-65',
                    7:'55-77',
                    8:'44-48',
                    9:'37-72',
                    10:'37-52',
                    11:'50-55',
                    12:'34-39',
                    13:'0-21',
                    14:'0-22',
                    15:'0-22',
                    16:'34-47',
                    17:'21-26',
                    18:'14-19',
                    19:'24-29',
                    20:'25-30',
                    21:'0-15,46-48',
                    22:'0-20'}

ALL_FILES = [
            "https://hicfiles.s3.amazonaws.com/hiseq/imr90/in-situ/combined.hic",
            "https://hicfiles.s3.amazonaws.com/hiseq/gm12878/agar/HIC030.hic",
            "https://hicfiles.s3.amazonaws.com/hiseq/gm12878/dilution/HIC034.hic",
            "https://hicfiles.s3.amazonaws.com/hiseq/gm12878/in-situ/combined.hic",
            "https://hicfiles.s3.amazonaws.com/hiseq/hmec/in-situ/combined.hic",
            "https://hicfiles.s3.amazonaws.com/hiseq/nhek/in-situ/combined.hic",
            "https://hicfiles.s3.amazonaws.com/hiseq/k562/in-situ/combined.hic",
            "https://hicfiles.s3.amazonaws.com/hiseq/kbm7/in-situ/combined.hic",
            "https://hicfiles.s3.amazonaws.com/hiseq/huvec/in-situ/combined.hic",
            "https://hicfiles.s3.amazonaws.com/hiseq/hela/in-situ/combined.hic",
            "https://hicfiles.s3.amazonaws.com/hiseq/hap1/in-situ/combined.hic"
            ]
            # "https://www.encodeproject.org/files/ENCFF675SJE/@@download/ENCFF675SJE.hic"  hg38 A549


GM12878_REPLICATES = [f'https://hicfiles.s3.amazonaws.com/hiseq/gm12878/in-situ/HIC00{i}.hic' for i in range(1, 9)]
# GM12878_REPLICATES.append(f'https://hicfiles.s3.amazonaws.com/hiseq/gm12878/in-situ/HIC010.hic')

HMEC_REPLICATES = [f'https://hicfiles.s3.amazonaws.com/hiseq/hmec/in-situ/HIC0{i}.hic' for i in range(58, 64)]



ALL_FILES_NO_GM12878 = [f for f in ALL_FILES if 'gm12878' not in f]

ALL_FILES_in_situ = [f for f in ALL_FILES if 'in-situ' in f]

VALIDATION_FILES = [
        "https://hicfiles.s3.amazonaws.com/hiseq/gm12878/in-situ/combined.hic",
        "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE104nnn/GSE104333/suppl/GSE104333_Rao-2017-untreated_combined_30.hic",
        "https://www.encodeproject.org/files/ENCFF177TYX/@@download/ENCFF177TYX.hic"
        ]
        # gm12878, HCT116, HL-60,

def intersect(region, bad_region):
    # region/2 is a tuple
    if region[0] >= bad_region[0] and region[0] < bad_region[1]:
        return True
    if region[1] > bad_region[0] and region[1] < bad_region[1]:
        return True
    if bad_region[0] >= region[0] and bad_region[0] < region[1]:
        return True

    return False
