# ------------------------------------------------------------------------------------- #
# This file takes the raw Tensorboard runs and concatenate them in a single usable file #
# ------------------------------------------------------------------------------------- #
# libraries
import matplotlib.pylab as plt
import pandas as pd
import numpy as np

# deeplab full autocorrelation run V4
deepFullRun_training = pd.read_csv(
    './paper_data/run-deeplab50_version_4-tag-training_loss.csv',
    usecols=[1, 2])
deepFullRun_validation = pd.read_csv(
    './paper_data/run-deeplab50_version_4-tag-val_loss.csv', usecols=[1, 2])
deepFullRun_lr = pd.read_csv(
    './paper_data/run-deeplab50_version_4-tag-lr-Adam.csv', usecols=[1, 2])
deepFullRun_radius = pd.read_csv(
    './paper_data/run-deeplab50_version_4-tag-mask_radius.csv', usecols=[1, 2])
deepFullRun_epoch = pd.read_csv(
    './paper_data/run-deeplab50_version_4-tag-epoch.csv', usecols=[1, 2])
deepFullRun_maxStep = deepFullRun_training['Step'].max()
deepFullRun_maxEpoch = deepFullRun_epoch['Value'].max()

# deeplab first shrink autocorrelation run V8
deepFirstShrinkRun_radius = pd.read_csv(
    './paper_data/run-deeplab50_version_8-tag-mask_radius.csv', usecols=[1, 2])
max_step = deepFirstShrinkRun_radius[deepFirstShrinkRun_radius['Value'] ==
                                     36.0]['Step'].max()
deepFirstShrinkRun_radius = pd.read_csv(
    './paper_data/run-deeplab50_version_8-tag-mask_radius.csv', usecols=[1, 2])
deepFirstShrinkRun_training = pd.read_csv(
    './paper_data/run-deeplab50_version_8-tag-training_loss.csv',
    usecols=[1, 2])
deepFirstShrinkRun_validation = pd.read_csv(
    './paper_data/run-deeplab50_version_8-tag-val_loss.csv', usecols=[1, 2])
deepFirstShrinkRun_lr = pd.read_csv(
    './paper_data/run-deeplab50_version_8-tag-lr-Adam.csv', usecols=[1, 2])
deepFirstShrinkRun_epoch = pd.read_csv(
    './paper_data/run-deeplab50_version_8-tag-epoch.csv', usecols=[1, 2])
deepFirstShrinkRun_radius = deepFirstShrinkRun_radius.loc[
    deepFirstShrinkRun_radius['Step'] <= max_step]
deepFirstShrinkRun_training = deepFirstShrinkRun_training.loc[
    deepFirstShrinkRun_training['Step'] <= max_step]
deepFirstShrinkRun_validation = deepFirstShrinkRun_validation.loc[
    deepFirstShrinkRun_validation['Step'] <= max_step]
deepFirstShrinkRun_lr = deepFirstShrinkRun_lr.loc[
    deepFirstShrinkRun_lr['Step'] <= max_step]
deepFirstShrinkRun_epoch = deepFirstShrinkRun_epoch.loc[
    deepFirstShrinkRun_epoch['Step'] <= max_step]
deepFirstShrinkRun_radius[
    'Step'] = deepFirstShrinkRun_radius['Step'] + deepFullRun_maxStep
deepFirstShrinkRun_training[
    'Step'] = deepFirstShrinkRun_training['Step'] + deepFullRun_maxStep
deepFirstShrinkRun_validation[
    'Step'] = deepFirstShrinkRun_validation['Step'] + deepFullRun_maxStep
deepFirstShrinkRun_lr[
    'Step'] = deepFirstShrinkRun_lr['Step'] + deepFullRun_maxStep
deepFirstShrinkRun_epoch[
    'Step'] = deepFirstShrinkRun_epoch['Step'] + deepFullRun_maxStep
deepFirstShrinkRun_epoch[
    'Value'] = deepFirstShrinkRun_epoch['Value'] + deepFullRun_maxEpoch
deepFirstShrinkRun_maxStep = deepFirstShrinkRun_training['Step'].max()
deepFirstShrinkRun_maxEpoch = deepFirstShrinkRun_epoch['Value'].max()

# deeplab second shrink autocorrelation run V9
deepSecondShrinkRun_radius = pd.read_csv(
    './paper_data/run-deeplab50_version_9-tag-mask_radius.csv', usecols=[1, 2])
max_step = deepSecondShrinkRun_radius[deepSecondShrinkRun_radius['Value'] ==
                                      31.0]['Step'].max()
deepSecondShrinkRun_radius = pd.read_csv(
    './paper_data/run-deeplab50_version_9-tag-mask_radius.csv', usecols=[1, 2])
deepSecondShrinkRun_training = pd.read_csv(
    './paper_data/run-deeplab50_version_9-tag-training_loss.csv',
    usecols=[1, 2])
deepSecondShrinkRun_validation = pd.read_csv(
    './paper_data/run-deeplab50_version_9-tag-val_loss.csv', usecols=[1, 2])
deepSecondShrinkRun_lr = pd.read_csv(
    './paper_data/run-deeplab50_version_9-tag-lr-Adam.csv', usecols=[1, 2])
deepSecondShrinkRun_epoch = pd.read_csv(
    './paper_data/run-deeplab50_version_9-tag-epoch.csv', usecols=[1, 2])
deepSecondShrinkRun_radius = deepSecondShrinkRun_radius.loc[
    deepSecondShrinkRun_radius['Step'] <= max_step]
deepSecondShrinkRun_training = deepSecondShrinkRun_training.loc[
    deepSecondShrinkRun_training['Step'] <= max_step]
deepSecondShrinkRun_validation = deepSecondShrinkRun_validation.loc[
    deepSecondShrinkRun_validation['Step'] <= max_step]
deepSecondShrinkRun_lr = deepSecondShrinkRun_lr.loc[
    deepSecondShrinkRun_lr['Step'] <= max_step]
deepSecondShrinkRun_epoch = deepSecondShrinkRun_epoch.loc[
    deepSecondShrinkRun_epoch['Step'] <= max_step]
deepSecondShrinkRun_radius[
    'Step'] = deepSecondShrinkRun_radius['Step'] + deepFirstShrinkRun_maxStep
deepSecondShrinkRun_training[
    'Step'] = deepSecondShrinkRun_training['Step'] + deepFirstShrinkRun_maxStep
deepSecondShrinkRun_validation['Step'] = deepSecondShrinkRun_validation[
    'Step'] + deepFirstShrinkRun_maxStep
deepSecondShrinkRun_lr[
    'Step'] = deepSecondShrinkRun_lr['Step'] + deepFirstShrinkRun_maxStep
deepSecondShrinkRun_epoch[
    'Step'] = deepSecondShrinkRun_epoch['Step'] + deepFirstShrinkRun_maxStep
deepSecondShrinkRun_epoch[
    'Value'] = deepSecondShrinkRun_epoch['Value'] + deepFirstShrinkRun_maxEpoch
deepSecondShrinkRun_maxStep = deepSecondShrinkRun_training['Step'].max()
deepSecondShrinkRun_maxEpoch = deepSecondShrinkRun_epoch['Value'].max()

# deeplab third shrink autocorrelation run V10
deepThirdShrinkRun_training = pd.read_csv(
    './paper_data/run-deeplab50_version_10-tag-training_loss.csv',
    usecols=[1, 2])
deepThirdShrinkRun_validation = pd.read_csv(
    './paper_data/run-deeplab50_version_10-tag-val_loss.csv', usecols=[1, 2])
deepThirdShrinkRun_lr = pd.read_csv(
    './paper_data/run-deeplab50_version_10-tag-lr-Adam.csv', usecols=[1, 2])
deepThirdShrinkRun_radius = pd.read_csv(
    './paper_data/run-deeplab50_version_10-tag-mask_radius.csv',
    usecols=[1, 2])
deepThirdShrinkRun_epoch = pd.read_csv(
    './paper_data/run-deeplab50_version_10-tag-epoch.csv', usecols=[1, 2])
deepThirdShrinkRun_radius[
    'Step'] = deepThirdShrinkRun_radius['Step'] + deepSecondShrinkRun_maxStep
deepThirdShrinkRun_training[
    'Step'] = deepThirdShrinkRun_training['Step'] + deepSecondShrinkRun_maxStep
deepThirdShrinkRun_validation['Step'] = deepThirdShrinkRun_validation[
    'Step'] + deepSecondShrinkRun_maxStep
deepThirdShrinkRun_lr[
    'Step'] = deepThirdShrinkRun_lr['Step'] + deepSecondShrinkRun_maxStep
deepThirdShrinkRun_epoch[
    'Step'] = deepThirdShrinkRun_epoch['Step'] + deepSecondShrinkRun_maxStep
deepThirdShrinkRun_epoch[
    'Value'] = deepThirdShrinkRun_epoch['Value'] + deepSecondShrinkRun_maxEpoch

# concatenate all dataframes
deep50_radius = pd.concat([
    deepFullRun_radius, deepFirstShrinkRun_radius, deepSecondShrinkRun_radius,
    deepThirdShrinkRun_radius
],
                          axis=0,
                          ignore_index=True)
deep50_training = pd.concat([
    deepFullRun_training, deepFirstShrinkRun_training,
    deepSecondShrinkRun_training, deepThirdShrinkRun_training
],
                            axis=0,
                            ignore_index=True)
deep50_validation = pd.concat([
    deepFullRun_validation, deepFirstShrinkRun_validation,
    deepSecondShrinkRun_validation, deepThirdShrinkRun_validation
],
                              axis=0,
                              ignore_index=True)
deep50_lr = pd.concat([
    deepFullRun_lr, deepFirstShrinkRun_lr, deepSecondShrinkRun_lr,
    deepThirdShrinkRun_lr
],
                      axis=0,
                      ignore_index=True)
deep50_epoch = pd.concat([
    deepFullRun_epoch, deepFirstShrinkRun_epoch, deepSecondShrinkRun_epoch,
    deepThirdShrinkRun_epoch
],
                         axis=0,
                         ignore_index=True)

# Save dataframes
deep50_radius.to_csv('./paper_data/deeplab50_radius.csv', index=False)
deep50_training.to_csv('./paper_data/deeplab50_training.csv', index=False)
deep50_validation.to_csv('./paper_data/deeplab50_validation.csv', index=False)
deep50_lr.to_csv('./paper_data/deeplab50_lr.csv', index=False)
deep50_epoch.to_csv('./paper_data/deeplab50_epoch.csv', index=False)
