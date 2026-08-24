set var IgJobReq.cmpJobID = 1
set var IgJobReq.numberOfProcessingPackets = 47
set var IgJobReq.firstPacketToProcess = 0
set var IgJobReq.cmpJobType = 8
set var IgJobReq.cmpJobDataType = 0
set var IgJobReq.cmp2dOverlapFlag = 0
set var IgJobReq.cmp3dOverlapFlag = 1
set var IgJobReq.cmpCtPetAlignmentFlag = 1
set var IgJobReq.breakPointFlag = 0
python _s("IgJobReq.inputEmissionFileName[0]", "/usr/PET/release/petig/selftest/data/selftest_kh3d_ex.rdf")
python _s("IgJobReq.inputEmissionFileName[1]", "")
python _s("IgJobReq.inputTransmissionFileName[0]", "/usr/PET/release/petig/selftest/data/selftest_kh3d_pifa.dat")
python _s("IgJobReq.inputTransmissionFileName[1]", "")
python _s("IgJobReq.normalizationSinogramFile", "/usr/PET/release/petig/selftest/data/selftest_kh3d_norm.rdf")
python _s("IgJobReq.blankscanSinogramFile", "")
python _s("IgJobReq.breakPointFile", "")
set var IgJobReq.outputDestination = 0
set var IgJobReq.emissionRandomsFlag = 3
set var IgJobReq.transmissionRandomsFlag = -1
set var IgJobReq.emissionDeadTimeFlag = 1
set var IgJobReq.transmissionDeadTimeFlag = 0
set var IgJobReq.normalizationFlag = 1
set var IgJobReq.blankscanFlag = 0
set var IgJobReq.naturalLogarithmFlag = 0
set var IgJobReq.radialRepositioningFlag = 0
set var IgJobReq.emissionScatterFlag = 2
set var IgJobReq.attenuationFlag = 2
set var IgJobReq.reconMethod = 2
set var IgJobReq.bpFilterFlag = 1
set var IgJobReq.bpFilterCutOff = 4.260000
set var IgJobReq.bpFilterOrder = 0
set var IgJobReq.bp3dFilterFlag = 3
set var IgJobReq.bp3dFilterFlagU = 3
set var IgJobReq.bp3dFilterCutOffU = 10.937500
set var IgJobReq.bp3dFilterOrderU = 0
set var IgJobReq.bp3dFilterFlagV = 1
set var IgJobReq.bp3dFilterCutOffV = 6.500000
set var IgJobReq.bp3dFilterOrderV = 0
set var IgJobReq.irNumSubsets = 24
set var IgJobReq.irNumIterations = 2
set var IgJobReq.irBpMatrixSize = 128
set var IgJobReq.irBpDFOV = 70.000000
set var IgJobReq.irBpImageCenterX = -0.000000
set var IgJobReq.irBpImageCenterY = 0.000000
set var IgJobReq.irLoopFilterFlag = 0
set var IgJobReq.irLoopRatioFilter = 0.000000
set var IgJobReq.irLoopCorrFilter = 0.000000
set var IgJobReq.irPostFilterFlag = 1
set var IgJobReq.irPostFilter = 6.000000
set var IgJobReq.irZAxisFilterFlag = 2
set var IgJobReq.bpMatrixSize = 256
set var IgJobReq.bpDFOV = 70.000000
set var IgJobReq.bpImageCenterX = 0.000000
set var IgJobReq.bpImageCenterY = 0.000000
set var IgJobReq.bp3dMatrixSizeXY = 128
set var IgJobReq.bp3dDFOV = 70.000000
set var IgJobReq.bp3dMatrixSizeZ = 47
set var IgJobReq.bp3dSpacingZ = 3.270000
set var IgJobReq.bp3dImageCenterX = -0.000000
set var IgJobReq.bp3dImageCenterY = 0.000000
set var IgJobReq.bp3dImageCenterZ = 0.000000
set var IgJobReq.decayFlag = 1
set var IgJobReq.decayTime = 0
set var IgJobReq.wellCounterCrossCalFlag = 2
set var IgJobReq.hrActivityFactor = 4.113164
set var IgJobReq.hsActivityFactor = 4.113164
set var IgJobReq.emissionSinoSmoothFlag = 0
set var IgJobReq.emissionSinoSmoothParam = 0.000000
set var IgJobReq.scatterParamHs[0] = -0.003174
set var IgJobReq.scatterParamHs[1] = -0.005350
set var IgJobReq.scatterParamHs[2] = 1.204000
set var IgJobReq.scatterParamHs[3] = -0.391900
set var IgJobReq.scatterParamHs[4] = -0.000006
set var IgJobReq.scatterParamHs[5] = -0.042700
set var IgJobReq.scatterParamHs[6] = -9.620000
set var IgJobReq.scatterParamHs[7] = -0.366000
set var IgJobReq.scatterParamHs[8] = 0.032100
set var IgJobReq.scatterParamHs[9] = 0.062100
set var IgJobReq.scatter3dPatientCutoff = 0.100000
set var IgJobReq.scatter3dPatientBackoff = 15.000000
set var IgJobReq.scatter3dMinimumField = 3
set var IgJobReq.scatter3dSmoothKernelSize = 3
set var IgJobReq.modelScatterIterations = 6
set var IgJobReq.modelScatterSliceWidth = 5.000000
set var IgJobReq.modelScatterMultiplesNorm = 0.330000
set var IgJobReq.modelScatterMultiplesWidth = 65.800003
set var IgJobReq.modelScatterMu = 0.000960
set var IgJobReq.foreWlsFilterCutoff = 24.000000
set var IgJobReq.macClipValue = 0.000912
set var IgJobReq.irMaxSubsetFlag = 2
set var IgJobReq.irExpandFOVFlag = 2
set var IgJobReq.irZAxisFilterRatio = 4.000000
set var IgJobReq.fCTACFlag = 1
set var IgJobReq.cmpCtPetTranslationFlag = 0
set var IgJobReq.cmpCtPetXTranslation = 0.000000
set var IgJobReq.cmpCtPetYTranslation = 0.000000
set var IgJobReq.cmpCtPetZTranslation = 0.000000
set var IgJobReq.imgFilter3dFlag = 0
set var IgJobReq.imgFilter3dWindowType = 0
set var IgJobReq.imgFilter3dWindowOrder = 0
set var IgJobReq.imgFilter3dCutOff = 0.000000
set var IgJobReq.cmpPackets[0].cmpProcessingPacketID = 0
set var IgJobReq.cmpPackets[0].sliceNumber[0] = 1
set var IgJobReq.cmpPackets[0].sliceNumber[1] = 0
set var IgJobReq.cmpPackets[0].sliceNumber[2] = 1
python _s("IgJobReq.cmpPackets[0].fileRead3dOverlap", "")
python _s("IgJobReq.cmpPackets[0].fileWrite3dOverlap", "/out/ovl/I1219182203.516963.F1219151167.851817.S1")
set var IgJobReq.cmpPackets[0].cmpPacketDataType = 0
set var IgJobReq.cmpPackets[0].wellCounterValue[0] = 0.990133
set var IgJobReq.cmpPackets[0].wellCounterValue[1] = 0.992431
set var IgJobReq.cmpPackets[1].cmpProcessingPacketID = 1
set var IgJobReq.cmpPackets[1].sliceNumber[0] = 2
set var IgJobReq.cmpPackets[1].sliceNumber[1] = 0
set var IgJobReq.cmpPackets[1].sliceNumber[2] = 2
python _s("IgJobReq.cmpPackets[1].fileRead3dOverlap", "")
python _s("IgJobReq.cmpPackets[1].fileWrite3dOverlap", "/out/ovl/I1219182203.516963.F1219151167.851817.S2")
set var IgJobReq.cmpPackets[1].cmpPacketDataType = 0
set var IgJobReq.cmpPackets[1].wellCounterValue[0] = 0.971399
set var IgJobReq.cmpPackets[1].wellCounterValue[1] = 1.010478
set var IgJobReq.cmpPackets[2].cmpProcessingPacketID = 2
set var IgJobReq.cmpPackets[2].sliceNumber[0] = 3
set var IgJobReq.cmpPackets[2].sliceNumber[1] = 0
set var IgJobReq.cmpPackets[2].sliceNumber[2] = 3
python _s("IgJobReq.cmpPackets[2].fileRead3dOverlap", "")
python _s("IgJobReq.cmpPackets[2].fileWrite3dOverlap", "/out/ovl/I1219182203.516963.F1219151167.851817.S3")
set var IgJobReq.cmpPackets[2].cmpPacketDataType = 0
set var IgJobReq.cmpPackets[2].wellCounterValue[0] = 0.961262
set var IgJobReq.cmpPackets[2].wellCounterValue[1] = 0.992563
set var IgJobReq.cmpPackets[3].cmpProcessingPacketID = 3
set var IgJobReq.cmpPackets[3].sliceNumber[0] = 4
set var IgJobReq.cmpPackets[3].sliceNumber[1] = 0
set var IgJobReq.cmpPackets[3].sliceNumber[2] = 4
python _s("IgJobReq.cmpPackets[3].fileRead3dOverlap", "")
python _s("IgJobReq.cmpPackets[3].fileWrite3dOverlap", "/out/ovl/I1219182203.516963.F1219151167.851817.S4")
set var IgJobReq.cmpPackets[3].cmpPacketDataType = 0
set var IgJobReq.cmpPackets[3].wellCounterValue[0] = 0.973307
set var IgJobReq.cmpPackets[3].wellCounterValue[1] = 1.017172
set var IgJobReq.cmpPackets[4].cmpProcessingPacketID = 4
set var IgJobReq.cmpPackets[4].sliceNumber[0] = 5
set var IgJobReq.cmpPackets[4].sliceNumber[1] = 0
set var IgJobReq.cmpPackets[4].sliceNumber[2] = 5
python _s("IgJobReq.cmpPackets[4].fileRead3dOverlap", "")
python _s("IgJobReq.cmpPackets[4].fileWrite3dOverlap", "/out/ovl/I1219182203.516963.F1219151167.851817.S5")
set var IgJobReq.cmpPackets[4].cmpPacketDataType = 0
set var IgJobReq.cmpPackets[4].wellCounterValue[0] = 0.979184
set var IgJobReq.cmpPackets[4].wellCounterValue[1] = 0.997921
set var IgJobReq.cmpPackets[5].cmpProcessingPacketID = 5
set var IgJobReq.cmpPackets[5].sliceNumber[0] = 6
set var IgJobReq.cmpPackets[5].sliceNumber[1] = 0
set var IgJobReq.cmpPackets[5].sliceNumber[2] = 6
python _s("IgJobReq.cmpPackets[5].fileRead3dOverlap", "")
python _s("IgJobReq.cmpPackets[5].fileWrite3dOverlap", "/out/ovl/I1219182203.516963.F1219151167.851817.S6")
set var IgJobReq.cmpPackets[5].cmpPacketDataType = 0
set var IgJobReq.cmpPackets[5].wellCounterValue[0] = 0.986909
set var IgJobReq.cmpPackets[5].wellCounterValue[1] = 1.014400
set var IgJobReq.cmpPackets[6].cmpProcessingPacketID = 6
set var IgJobReq.cmpPackets[6].sliceNumber[0] = 7
set var IgJobReq.cmpPackets[6].sliceNumber[1] = 0
set var IgJobReq.cmpPackets[6].sliceNumber[2] = 7
python _s("IgJobReq.cmpPackets[6].fileRead3dOverlap", "")
python _s("IgJobReq.cmpPackets[6].fileWrite3dOverlap", "/out/ovl/I1219182203.516963.F1219151167.851817.S7")
set var IgJobReq.cmpPackets[6].cmpPacketDataType = 0
set var IgJobReq.cmpPackets[6].wellCounterValue[0] = 1.002100
set var IgJobReq.cmpPackets[6].wellCounterValue[1] = 1.000013
set var IgJobReq.cmpPackets[7].cmpProcessingPacketID = 7
set var IgJobReq.cmpPackets[7].sliceNumber[0] = 8
set var IgJobReq.cmpPackets[7].sliceNumber[1] = 0
set var IgJobReq.cmpPackets[7].sliceNumber[2] = 8
python _s("IgJobReq.cmpPackets[7].fileRead3dOverlap", "")
python _s("IgJobReq.cmpPackets[7].fileWrite3dOverlap", "/out/ovl/I1219182203.516963.F1219151167.851817.S8")
set var IgJobReq.cmpPackets[7].cmpPacketDataType = 0
set var IgJobReq.cmpPackets[7].wellCounterValue[0] = 0.995041
set var IgJobReq.cmpPackets[7].wellCounterValue[1] = 1.012942
set var IgJobReq.cmpPackets[8].cmpProcessingPacketID = 8
set var IgJobReq.cmpPackets[8].sliceNumber[0] = 9
set var IgJobReq.cmpPackets[8].sliceNumber[1] = 0
set var IgJobReq.cmpPackets[8].sliceNumber[2] = 9
python _s("IgJobReq.cmpPackets[8].fileRead3dOverlap", "")
python _s("IgJobReq.cmpPackets[8].fileWrite3dOverlap", "/out/ovl/I1219182203.516963.F1219151167.851817.S9")
set var IgJobReq.cmpPackets[8].cmpPacketDataType = 0
set var IgJobReq.cmpPackets[8].wellCounterValue[0] = 1.001218
set var IgJobReq.cmpPackets[8].wellCounterValue[1] = 1.003055
set var IgJobReq.cmpPackets[9].cmpProcessingPacketID = 9
set var IgJobReq.cmpPackets[9].sliceNumber[0] = 10
set var IgJobReq.cmpPackets[9].sliceNumber[1] = 0
set var IgJobReq.cmpPackets[9].sliceNumber[2] = 10
python _s("IgJobReq.cmpPackets[9].fileRead3dOverlap", "")
python _s("IgJobReq.cmpPackets[9].fileWrite3dOverlap", "/out/ovl/I1219182203.516963.F1219151167.851817.S10")
set var IgJobReq.cmpPackets[9].cmpPacketDataType = 0
set var IgJobReq.cmpPackets[9].wellCounterValue[0] = 1.003948
set var IgJobReq.cmpPackets[9].wellCounterValue[1] = 1.021917
set var IgJobReq.cmpPackets[10].cmpProcessingPacketID = 10
set var IgJobReq.cmpPackets[10].sliceNumber[0] = 11
set var IgJobReq.cmpPackets[10].sliceNumber[1] = 0
set var IgJobReq.cmpPackets[10].sliceNumber[2] = 11
python _s("IgJobReq.cmpPackets[10].fileRead3dOverlap", "")
python _s("IgJobReq.cmpPackets[10].fileWrite3dOverlap", "/out/ovl/I1219182203.516963.F1219151167.851817.S11")
set var IgJobReq.cmpPackets[10].cmpPacketDataType = 0
set var IgJobReq.cmpPackets[10].wellCounterValue[0] = 0.998041
set var IgJobReq.cmpPackets[10].wellCounterValue[1] = 1.011541
set var IgJobReq.cmpPackets[11].cmpProcessingPacketID = 11
set var IgJobReq.cmpPackets[11].sliceNumber[0] = 12
set var IgJobReq.cmpPackets[11].sliceNumber[1] = 0
set var IgJobReq.cmpPackets[11].sliceNumber[2] = 12
python _s("IgJobReq.cmpPackets[11].fileRead3dOverlap", "")
python _s("IgJobReq.cmpPackets[11].fileWrite3dOverlap", "/out/ovl/I1219182203.516963.F1219151167.851817.S12")
set var IgJobReq.cmpPackets[11].cmpPacketDataType = 0
set var IgJobReq.cmpPackets[11].wellCounterValue[0] = 1.002490
set var IgJobReq.cmpPackets[11].wellCounterValue[1] = 1.029799
set var IgJobReq.cmpPackets[12].cmpProcessingPacketID = 12
set var IgJobReq.cmpPackets[12].sliceNumber[0] = 13
set var IgJobReq.cmpPackets[12].sliceNumber[1] = 0
set var IgJobReq.cmpPackets[12].sliceNumber[2] = 13
python _s("IgJobReq.cmpPackets[12].fileRead3dOverlap", "")
python _s("IgJobReq.cmpPackets[12].fileWrite3dOverlap", "/out/ovl/I1219182203.516963.F1219151167.851817.S13")
set var IgJobReq.cmpPackets[12].cmpPacketDataType = 0
set var IgJobReq.cmpPackets[12].wellCounterValue[0] = 0.994143
set var IgJobReq.cmpPackets[12].wellCounterValue[1] = 1.021096
set var IgJobReq.cmpPackets[13].cmpProcessingPacketID = 13
set var IgJobReq.cmpPackets[13].sliceNumber[0] = 14
set var IgJobReq.cmpPackets[13].sliceNumber[1] = 0
set var IgJobReq.cmpPackets[13].sliceNumber[2] = 14
python _s("IgJobReq.cmpPackets[13].fileRead3dOverlap", "")
python _s("IgJobReq.cmpPackets[13].fileWrite3dOverlap", "/out/ovl/I1219182203.516963.F1219151167.851817.S14")
set var IgJobReq.cmpPackets[13].cmpPacketDataType = 0
set var IgJobReq.cmpPackets[13].wellCounterValue[0] = 1.001569
set var IgJobReq.cmpPackets[13].wellCounterValue[1] = 1.018991
set var IgJobReq.cmpPackets[14].cmpProcessingPacketID = 14
set var IgJobReq.cmpPackets[14].sliceNumber[0] = 15
set var IgJobReq.cmpPackets[14].sliceNumber[1] = 0
set var IgJobReq.cmpPackets[14].sliceNumber[2] = 15
python _s("IgJobReq.cmpPackets[14].fileRead3dOverlap", "")
python _s("IgJobReq.cmpPackets[14].fileWrite3dOverlap", "/out/ovl/I1219182203.516963.F1219151167.851817.S15")
set var IgJobReq.cmpPackets[14].cmpPacketDataType = 0
set var IgJobReq.cmpPackets[14].wellCounterValue[0] = 1.006124
set var IgJobReq.cmpPackets[14].wellCounterValue[1] = 1.016524
set var IgJobReq.cmpPackets[15].cmpProcessingPacketID = 15
set var IgJobReq.cmpPackets[15].sliceNumber[0] = 16
set var IgJobReq.cmpPackets[15].sliceNumber[1] = 0
set var IgJobReq.cmpPackets[15].sliceNumber[2] = 16
python _s("IgJobReq.cmpPackets[15].fileRead3dOverlap", "")
python _s("IgJobReq.cmpPackets[15].fileWrite3dOverlap", "/out/ovl/I1219182203.516963.F1219151167.851817.S16")
set var IgJobReq.cmpPackets[15].cmpPacketDataType = 0
set var IgJobReq.cmpPackets[15].wellCounterValue[0] = 1.006796
set var IgJobReq.cmpPackets[15].wellCounterValue[1] = 1.008708
set var IgJobReq.cmpPackets[16].cmpProcessingPacketID = 16
set var IgJobReq.cmpPackets[16].sliceNumber[0] = 17
set var IgJobReq.cmpPackets[16].sliceNumber[1] = 0
set var IgJobReq.cmpPackets[16].sliceNumber[2] = 17
python _s("IgJobReq.cmpPackets[16].fileRead3dOverlap", "")
python _s("IgJobReq.cmpPackets[16].fileWrite3dOverlap", "/out/ovl/I1219182203.516963.F1219151167.851817.S17")
set var IgJobReq.cmpPackets[16].cmpPacketDataType = 0
set var IgJobReq.cmpPackets[16].wellCounterValue[0] = 1.016977
set var IgJobReq.cmpPackets[16].wellCounterValue[1] = 1.005449
set var IgJobReq.cmpPackets[17].cmpProcessingPacketID = 17
set var IgJobReq.cmpPackets[17].sliceNumber[0] = 18
set var IgJobReq.cmpPackets[17].sliceNumber[1] = 0
set var IgJobReq.cmpPackets[17].sliceNumber[2] = 18
python _s("IgJobReq.cmpPackets[17].fileRead3dOverlap", "")
python _s("IgJobReq.cmpPackets[17].fileWrite3dOverlap", "/out/ovl/I1219182203.516963.F1219151167.851817.S18")
set var IgJobReq.cmpPackets[17].cmpPacketDataType = 0
set var IgJobReq.cmpPackets[17].wellCounterValue[0] = 1.009912
set var IgJobReq.cmpPackets[17].wellCounterValue[1] = 0.992626
set var IgJobReq.cmpPackets[18].cmpProcessingPacketID = 18
set var IgJobReq.cmpPackets[18].sliceNumber[0] = 19
set var IgJobReq.cmpPackets[18].sliceNumber[1] = 0
set var IgJobReq.cmpPackets[18].sliceNumber[2] = 19
python _s("IgJobReq.cmpPackets[18].fileRead3dOverlap", "")
python _s("IgJobReq.cmpPackets[18].fileWrite3dOverlap", "/out/ovl/I1219182203.516963.F1219151167.851817.S19")
set var IgJobReq.cmpPackets[18].cmpPacketDataType = 0
set var IgJobReq.cmpPackets[18].wellCounterValue[0] = 1.011456
set var IgJobReq.cmpPackets[18].wellCounterValue[1] = 0.986773
set var IgJobReq.cmpPackets[19].cmpProcessingPacketID = 19
set var IgJobReq.cmpPackets[19].sliceNumber[0] = 20
set var IgJobReq.cmpPackets[19].sliceNumber[1] = 0
set var IgJobReq.cmpPackets[19].sliceNumber[2] = 20
python _s("IgJobReq.cmpPackets[19].fileRead3dOverlap", "")
python _s("IgJobReq.cmpPackets[19].fileWrite3dOverlap", "/out/ovl/I1219182203.516963.F1219151167.851817.S20")
set var IgJobReq.cmpPackets[19].cmpPacketDataType = 0
set var IgJobReq.cmpPackets[19].wellCounterValue[0] = 1.001414
set var IgJobReq.cmpPackets[19].wellCounterValue[1] = 0.967239
set var IgJobReq.cmpPackets[20].cmpProcessingPacketID = 20
set var IgJobReq.cmpPackets[20].sliceNumber[0] = 21
set var IgJobReq.cmpPackets[20].sliceNumber[1] = 0
set var IgJobReq.cmpPackets[20].sliceNumber[2] = 21
python _s("IgJobReq.cmpPackets[20].fileRead3dOverlap", "")
python _s("IgJobReq.cmpPackets[20].fileWrite3dOverlap", "/out/ovl/I1219182203.516963.F1219151167.851817.S21")
set var IgJobReq.cmpPackets[20].cmpPacketDataType = 0
set var IgJobReq.cmpPackets[20].wellCounterValue[0] = 0.998997
set var IgJobReq.cmpPackets[20].wellCounterValue[1] = 0.976526
set var IgJobReq.cmpPackets[21].cmpProcessingPacketID = 21
set var IgJobReq.cmpPackets[21].sliceNumber[0] = 22
set var IgJobReq.cmpPackets[21].sliceNumber[1] = 0
set var IgJobReq.cmpPackets[21].sliceNumber[2] = 22
python _s("IgJobReq.cmpPackets[21].fileRead3dOverlap", "")
python _s("IgJobReq.cmpPackets[21].fileWrite3dOverlap", "/out/ovl/I1219182203.516963.F1219151167.851817.S22")
set var IgJobReq.cmpPackets[21].cmpPacketDataType = 0
set var IgJobReq.cmpPackets[21].wellCounterValue[0] = 1.002006
set var IgJobReq.cmpPackets[21].wellCounterValue[1] = 0.977898
set var IgJobReq.cmpPackets[22].cmpProcessingPacketID = 22
set var IgJobReq.cmpPackets[22].sliceNumber[0] = 23
set var IgJobReq.cmpPackets[22].sliceNumber[1] = 0
set var IgJobReq.cmpPackets[22].sliceNumber[2] = 23
python _s("IgJobReq.cmpPackets[22].fileRead3dOverlap", "")
python _s("IgJobReq.cmpPackets[22].fileWrite3dOverlap", "/out/ovl/I1219182203.516963.F1219151167.851817.S23")
set var IgJobReq.cmpPackets[22].cmpPacketDataType = 0
set var IgJobReq.cmpPackets[22].wellCounterValue[0] = 0.991765
set var IgJobReq.cmpPackets[22].wellCounterValue[1] = 1.009658
set var IgJobReq.cmpPackets[23].cmpProcessingPacketID = 23
set var IgJobReq.cmpPackets[23].sliceNumber[0] = 24
set var IgJobReq.cmpPackets[23].sliceNumber[1] = 0
set var IgJobReq.cmpPackets[23].sliceNumber[2] = 0
python _s("IgJobReq.cmpPackets[23].fileRead3dOverlap", "")
python _s("IgJobReq.cmpPackets[23].fileWrite3dOverlap", "")
set var IgJobReq.cmpPackets[23].cmpPacketDataType = 0
set var IgJobReq.cmpPackets[23].wellCounterValue[0] = 1.008092
set var IgJobReq.cmpPackets[23].wellCounterValue[1] = 0.000000
set var IgJobReq.cmpPackets[24].cmpProcessingPacketID = 24
set var IgJobReq.cmpPackets[24].sliceNumber[0] = 25
set var IgJobReq.cmpPackets[24].sliceNumber[1] = 0
set var IgJobReq.cmpPackets[24].sliceNumber[2] = 0
python _s("IgJobReq.cmpPackets[24].fileRead3dOverlap", "")
python _s("IgJobReq.cmpPackets[24].fileWrite3dOverlap", "")
set var IgJobReq.cmpPackets[24].cmpPacketDataType = 0
set var IgJobReq.cmpPackets[24].wellCounterValue[0] = 0.992431
set var IgJobReq.cmpPackets[24].wellCounterValue[1] = 0.000000
set var IgJobReq.cmpPackets[25].cmpProcessingPacketID = 25
set var IgJobReq.cmpPackets[25].sliceNumber[0] = 26
set var IgJobReq.cmpPackets[25].sliceNumber[1] = 0
set var IgJobReq.cmpPackets[25].sliceNumber[2] = 0
python _s("IgJobReq.cmpPackets[25].fileRead3dOverlap", "")
python _s("IgJobReq.cmpPackets[25].fileWrite3dOverlap", "")
set var IgJobReq.cmpPackets[25].cmpPacketDataType = 0
set var IgJobReq.cmpPackets[25].wellCounterValue[0] = 1.010478
set var IgJobReq.cmpPackets[25].wellCounterValue[1] = 0.000000
set var IgJobReq.cmpPackets[26].cmpProcessingPacketID = 26
set var IgJobReq.cmpPackets[26].sliceNumber[0] = 27
set var IgJobReq.cmpPackets[26].sliceNumber[1] = 0
set var IgJobReq.cmpPackets[26].sliceNumber[2] = 0
python _s("IgJobReq.cmpPackets[26].fileRead3dOverlap", "")
python _s("IgJobReq.cmpPackets[26].fileWrite3dOverlap", "")
set var IgJobReq.cmpPackets[26].cmpPacketDataType = 0
set var IgJobReq.cmpPackets[26].wellCounterValue[0] = 0.992563
set var IgJobReq.cmpPackets[26].wellCounterValue[1] = 0.000000
set var IgJobReq.cmpPackets[27].cmpProcessingPacketID = 27
set var IgJobReq.cmpPackets[27].sliceNumber[0] = 28
set var IgJobReq.cmpPackets[27].sliceNumber[1] = 0
set var IgJobReq.cmpPackets[27].sliceNumber[2] = 0
python _s("IgJobReq.cmpPackets[27].fileRead3dOverlap", "")
python _s("IgJobReq.cmpPackets[27].fileWrite3dOverlap", "")
set var IgJobReq.cmpPackets[27].cmpPacketDataType = 0
set var IgJobReq.cmpPackets[27].wellCounterValue[0] = 1.017172
set var IgJobReq.cmpPackets[27].wellCounterValue[1] = 0.000000
set var IgJobReq.cmpPackets[28].cmpProcessingPacketID = 28
set var IgJobReq.cmpPackets[28].sliceNumber[0] = 29
set var IgJobReq.cmpPackets[28].sliceNumber[1] = 0
set var IgJobReq.cmpPackets[28].sliceNumber[2] = 0
python _s("IgJobReq.cmpPackets[28].fileRead3dOverlap", "")
python _s("IgJobReq.cmpPackets[28].fileWrite3dOverlap", "")
set var IgJobReq.cmpPackets[28].cmpPacketDataType = 0
set var IgJobReq.cmpPackets[28].wellCounterValue[0] = 0.997921
set var IgJobReq.cmpPackets[28].wellCounterValue[1] = 0.000000
set var IgJobReq.cmpPackets[29].cmpProcessingPacketID = 29
set var IgJobReq.cmpPackets[29].sliceNumber[0] = 30
set var IgJobReq.cmpPackets[29].sliceNumber[1] = 0
set var IgJobReq.cmpPackets[29].sliceNumber[2] = 0
python _s("IgJobReq.cmpPackets[29].fileRead3dOverlap", "")
python _s("IgJobReq.cmpPackets[29].fileWrite3dOverlap", "")
set var IgJobReq.cmpPackets[29].cmpPacketDataType = 0
set var IgJobReq.cmpPackets[29].wellCounterValue[0] = 1.014400
set var IgJobReq.cmpPackets[29].wellCounterValue[1] = 0.000000
set var IgJobReq.cmpPackets[30].cmpProcessingPacketID = 30
set var IgJobReq.cmpPackets[30].sliceNumber[0] = 31
set var IgJobReq.cmpPackets[30].sliceNumber[1] = 0
set var IgJobReq.cmpPackets[30].sliceNumber[2] = 0
python _s("IgJobReq.cmpPackets[30].fileRead3dOverlap", "")
python _s("IgJobReq.cmpPackets[30].fileWrite3dOverlap", "")
set var IgJobReq.cmpPackets[30].cmpPacketDataType = 0
set var IgJobReq.cmpPackets[30].wellCounterValue[0] = 1.000013
set var IgJobReq.cmpPackets[30].wellCounterValue[1] = 0.000000
set var IgJobReq.cmpPackets[31].cmpProcessingPacketID = 31
set var IgJobReq.cmpPackets[31].sliceNumber[0] = 32
set var IgJobReq.cmpPackets[31].sliceNumber[1] = 0
set var IgJobReq.cmpPackets[31].sliceNumber[2] = 0
python _s("IgJobReq.cmpPackets[31].fileRead3dOverlap", "")
python _s("IgJobReq.cmpPackets[31].fileWrite3dOverlap", "")
set var IgJobReq.cmpPackets[31].cmpPacketDataType = 0
set var IgJobReq.cmpPackets[31].wellCounterValue[0] = 1.012942
set var IgJobReq.cmpPackets[31].wellCounterValue[1] = 0.000000
set var IgJobReq.cmpPackets[32].cmpProcessingPacketID = 32
set var IgJobReq.cmpPackets[32].sliceNumber[0] = 33
set var IgJobReq.cmpPackets[32].sliceNumber[1] = 0
set var IgJobReq.cmpPackets[32].sliceNumber[2] = 0
python _s("IgJobReq.cmpPackets[32].fileRead3dOverlap", "")
python _s("IgJobReq.cmpPackets[32].fileWrite3dOverlap", "")
set var IgJobReq.cmpPackets[32].cmpPacketDataType = 0
set var IgJobReq.cmpPackets[32].wellCounterValue[0] = 1.003055
set var IgJobReq.cmpPackets[32].wellCounterValue[1] = 0.000000
set var IgJobReq.cmpPackets[33].cmpProcessingPacketID = 33
set var IgJobReq.cmpPackets[33].sliceNumber[0] = 34
set var IgJobReq.cmpPackets[33].sliceNumber[1] = 0
set var IgJobReq.cmpPackets[33].sliceNumber[2] = 0
python _s("IgJobReq.cmpPackets[33].fileRead3dOverlap", "")
python _s("IgJobReq.cmpPackets[33].fileWrite3dOverlap", "")
set var IgJobReq.cmpPackets[33].cmpPacketDataType = 0
set var IgJobReq.cmpPackets[33].wellCounterValue[0] = 1.021917
set var IgJobReq.cmpPackets[33].wellCounterValue[1] = 0.000000
set var IgJobReq.cmpPackets[34].cmpProcessingPacketID = 34
set var IgJobReq.cmpPackets[34].sliceNumber[0] = 35
set var IgJobReq.cmpPackets[34].sliceNumber[1] = 0
set var IgJobReq.cmpPackets[34].sliceNumber[2] = 0
python _s("IgJobReq.cmpPackets[34].fileRead3dOverlap", "")
python _s("IgJobReq.cmpPackets[34].fileWrite3dOverlap", "")
set var IgJobReq.cmpPackets[34].cmpPacketDataType = 0
set var IgJobReq.cmpPackets[34].wellCounterValue[0] = 1.011541
set var IgJobReq.cmpPackets[34].wellCounterValue[1] = 0.000000
set var IgJobReq.cmpPackets[35].cmpProcessingPacketID = 35
set var IgJobReq.cmpPackets[35].sliceNumber[0] = 36
set var IgJobReq.cmpPackets[35].sliceNumber[1] = 0
set var IgJobReq.cmpPackets[35].sliceNumber[2] = 0
python _s("IgJobReq.cmpPackets[35].fileRead3dOverlap", "")
python _s("IgJobReq.cmpPackets[35].fileWrite3dOverlap", "")
set var IgJobReq.cmpPackets[35].cmpPacketDataType = 0
set var IgJobReq.cmpPackets[35].wellCounterValue[0] = 1.029799
set var IgJobReq.cmpPackets[35].wellCounterValue[1] = 0.000000
set var IgJobReq.cmpPackets[36].cmpProcessingPacketID = 36
set var IgJobReq.cmpPackets[36].sliceNumber[0] = 37
set var IgJobReq.cmpPackets[36].sliceNumber[1] = 0
set var IgJobReq.cmpPackets[36].sliceNumber[2] = 0
python _s("IgJobReq.cmpPackets[36].fileRead3dOverlap", "")
python _s("IgJobReq.cmpPackets[36].fileWrite3dOverlap", "")
set var IgJobReq.cmpPackets[36].cmpPacketDataType = 0
set var IgJobReq.cmpPackets[36].wellCounterValue[0] = 1.021096
set var IgJobReq.cmpPackets[36].wellCounterValue[1] = 0.000000
set var IgJobReq.cmpPackets[37].cmpProcessingPacketID = 37
set var IgJobReq.cmpPackets[37].sliceNumber[0] = 38
set var IgJobReq.cmpPackets[37].sliceNumber[1] = 0
set var IgJobReq.cmpPackets[37].sliceNumber[2] = 0
python _s("IgJobReq.cmpPackets[37].fileRead3dOverlap", "")
python _s("IgJobReq.cmpPackets[37].fileWrite3dOverlap", "")
set var IgJobReq.cmpPackets[37].cmpPacketDataType = 0
set var IgJobReq.cmpPackets[37].wellCounterValue[0] = 1.018991
set var IgJobReq.cmpPackets[37].wellCounterValue[1] = 0.000000
set var IgJobReq.cmpPackets[38].cmpProcessingPacketID = 38
set var IgJobReq.cmpPackets[38].sliceNumber[0] = 39
set var IgJobReq.cmpPackets[38].sliceNumber[1] = 0
set var IgJobReq.cmpPackets[38].sliceNumber[2] = 0
python _s("IgJobReq.cmpPackets[38].fileRead3dOverlap", "")
python _s("IgJobReq.cmpPackets[38].fileWrite3dOverlap", "")
set var IgJobReq.cmpPackets[38].cmpPacketDataType = 0
set var IgJobReq.cmpPackets[38].wellCounterValue[0] = 1.016524
set var IgJobReq.cmpPackets[38].wellCounterValue[1] = 0.000000
set var IgJobReq.cmpPackets[39].cmpProcessingPacketID = 39
set var IgJobReq.cmpPackets[39].sliceNumber[0] = 40
set var IgJobReq.cmpPackets[39].sliceNumber[1] = 0
set var IgJobReq.cmpPackets[39].sliceNumber[2] = 0
python _s("IgJobReq.cmpPackets[39].fileRead3dOverlap", "")
python _s("IgJobReq.cmpPackets[39].fileWrite3dOverlap", "")
set var IgJobReq.cmpPackets[39].cmpPacketDataType = 0
set var IgJobReq.cmpPackets[39].wellCounterValue[0] = 1.008708
set var IgJobReq.cmpPackets[39].wellCounterValue[1] = 0.000000
set var IgJobReq.cmpPackets[40].cmpProcessingPacketID = 40
set var IgJobReq.cmpPackets[40].sliceNumber[0] = 41
set var IgJobReq.cmpPackets[40].sliceNumber[1] = 0
set var IgJobReq.cmpPackets[40].sliceNumber[2] = 0
python _s("IgJobReq.cmpPackets[40].fileRead3dOverlap", "")
python _s("IgJobReq.cmpPackets[40].fileWrite3dOverlap", "")
set var IgJobReq.cmpPackets[40].cmpPacketDataType = 0
set var IgJobReq.cmpPackets[40].wellCounterValue[0] = 1.005449
set var IgJobReq.cmpPackets[40].wellCounterValue[1] = 0.000000
set var IgJobReq.cmpPackets[41].cmpProcessingPacketID = 41
set var IgJobReq.cmpPackets[41].sliceNumber[0] = 42
set var IgJobReq.cmpPackets[41].sliceNumber[1] = 0
set var IgJobReq.cmpPackets[41].sliceNumber[2] = 0
python _s("IgJobReq.cmpPackets[41].fileRead3dOverlap", "")
python _s("IgJobReq.cmpPackets[41].fileWrite3dOverlap", "")
set var IgJobReq.cmpPackets[41].cmpPacketDataType = 0
set var IgJobReq.cmpPackets[41].wellCounterValue[0] = 0.992626
set var IgJobReq.cmpPackets[41].wellCounterValue[1] = 0.000000
set var IgJobReq.cmpPackets[42].cmpProcessingPacketID = 42
set var IgJobReq.cmpPackets[42].sliceNumber[0] = 43
set var IgJobReq.cmpPackets[42].sliceNumber[1] = 0
set var IgJobReq.cmpPackets[42].sliceNumber[2] = 0
python _s("IgJobReq.cmpPackets[42].fileRead3dOverlap", "")
python _s("IgJobReq.cmpPackets[42].fileWrite3dOverlap", "")
set var IgJobReq.cmpPackets[42].cmpPacketDataType = 0
set var IgJobReq.cmpPackets[42].wellCounterValue[0] = 0.986773
set var IgJobReq.cmpPackets[42].wellCounterValue[1] = 0.000000
set var IgJobReq.cmpPackets[43].cmpProcessingPacketID = 43
set var IgJobReq.cmpPackets[43].sliceNumber[0] = 44
set var IgJobReq.cmpPackets[43].sliceNumber[1] = 0
set var IgJobReq.cmpPackets[43].sliceNumber[2] = 0
python _s("IgJobReq.cmpPackets[43].fileRead3dOverlap", "")
python _s("IgJobReq.cmpPackets[43].fileWrite3dOverlap", "")
set var IgJobReq.cmpPackets[43].cmpPacketDataType = 0
set var IgJobReq.cmpPackets[43].wellCounterValue[0] = 0.967239
set var IgJobReq.cmpPackets[43].wellCounterValue[1] = 0.000000
set var IgJobReq.cmpPackets[44].cmpProcessingPacketID = 44
set var IgJobReq.cmpPackets[44].sliceNumber[0] = 45
set var IgJobReq.cmpPackets[44].sliceNumber[1] = 0
set var IgJobReq.cmpPackets[44].sliceNumber[2] = 0
python _s("IgJobReq.cmpPackets[44].fileRead3dOverlap", "")
python _s("IgJobReq.cmpPackets[44].fileWrite3dOverlap", "")
set var IgJobReq.cmpPackets[44].cmpPacketDataType = 0
set var IgJobReq.cmpPackets[44].wellCounterValue[0] = 0.976526
set var IgJobReq.cmpPackets[44].wellCounterValue[1] = 0.000000
set var IgJobReq.cmpPackets[45].cmpProcessingPacketID = 45
set var IgJobReq.cmpPackets[45].sliceNumber[0] = 46
set var IgJobReq.cmpPackets[45].sliceNumber[1] = 0
set var IgJobReq.cmpPackets[45].sliceNumber[2] = 0
python _s("IgJobReq.cmpPackets[45].fileRead3dOverlap", "")
python _s("IgJobReq.cmpPackets[45].fileWrite3dOverlap", "")
set var IgJobReq.cmpPackets[45].cmpPacketDataType = 0
set var IgJobReq.cmpPackets[45].wellCounterValue[0] = 0.977898
set var IgJobReq.cmpPackets[45].wellCounterValue[1] = 0.000000
set var IgJobReq.cmpPackets[46].cmpProcessingPacketID = 46
set var IgJobReq.cmpPackets[46].sliceNumber[0] = 47
set var IgJobReq.cmpPackets[46].sliceNumber[1] = 0
set var IgJobReq.cmpPackets[46].sliceNumber[2] = 0
python _s("IgJobReq.cmpPackets[46].fileRead3dOverlap", "")
python _s("IgJobReq.cmpPackets[46].fileWrite3dOverlap", "")
set var IgJobReq.cmpPackets[46].cmpPacketDataType = 0
set var IgJobReq.cmpPackets[46].wellCounterValue[0] = 1.009658
set var IgJobReq.cmpPackets[46].wellCounterValue[1] = 0.000000
