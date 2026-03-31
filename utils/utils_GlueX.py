import numpy as np

pixel_width = 6.
pixel_height = 6.

def convert_indices_gt(pmtID,pixelID): 
    pmtID = np.array(pmtID).astype("float32")
    o_box = pmtID//108
    if o_box[0] == 1:
        pmtID -= 108

    pixelID = np.array(pixelID).astype('float32')
    row = (pmtID//18) * 8 + pixelID//8
    col = (pmtID%18) * 8 + pixelID%8

    x = 2 + col * 6. + (pmtID % 18) * 2. + 3.
    y = 2 + row * 6. + (pmtID // 18) * 2. + 3.

    return x,y


bins_x = [0,5.-pixel_width/2,    5. + pixel_width/2, 11.+pixel_width/2,  17.+pixel_width/2,  23.+pixel_width/2, 
             29.+pixel_width/2,  35+pixel_width/2.,  41.+pixel_width/2,  47.+pixel_width/2,  55. - pixel_width/2, 55. +pixel_width/2, # PMT 1 
             61.+pixel_width/2,  67.+pixel_width/2,  73.+pixel_width/2,  79.+pixel_width/2,  85.+pixel_width/2,
             91.+pixel_width/2,  97.+pixel_width/2,  105.-pixel_width/2, 105.+pixel_width/2, # PMT 2
             111.+pixel_width/2, 117.+pixel_width/2, 123.+pixel_width/2, 129.+pixel_width/2, 135.+pixel_width/2,
             141.+pixel_width/2, 147.+pixel_width/2, 155.-pixel_width/2, 155.+pixel_width/2, # PMT 3
             161.+pixel_width/2, 167.+pixel_width/2, 173.+pixel_width/2, 179.+pixel_width/2, 185.+pixel_width/2,
             191.+pixel_width/2, 197.+pixel_width/2, 205.-pixel_width/2, 205.+pixel_width/2, # PMT 4
             211.+pixel_width/2, 217.+pixel_width/2, 223.+pixel_width/2, 229.+pixel_width/2, 235.+pixel_width/2,
             241.+pixel_width/2, 247.+pixel_width/2, 255.-pixel_width/2, 255.+pixel_width/2, # PMT 5
             261.+pixel_width/2, 267.+pixel_width/2, 273.+pixel_width/2, 279.+pixel_width/2, 285.+pixel_width/2,
             291.+pixel_width/2, 297.+pixel_width/2, 305.-pixel_width/2, 305+pixel_width/2, # PMT 6
             311.+pixel_width/2, 317.+pixel_width/2, 323.+pixel_width/2, 329.+pixel_width/2, 335.+pixel_width/2,
             341.+pixel_width/2, 347.+pixel_width/2, 355.-pixel_width/2, 355.+pixel_width/2, # PMT 7
             361.+pixel_width/2, 367.+pixel_width/2, 373.+pixel_width/2, 379.+pixel_width/2, 385.+pixel_width/2,
             391.+pixel_width/2, 397.+pixel_width/2, 405.-pixel_width/2, 405.+pixel_width/2, # PMT 8
             411.+pixel_width/2, 417.+pixel_width/2, 423.+pixel_width/2, 429.+pixel_width/2, 435.+pixel_width/2,
             441.+pixel_width/2, 447.+pixel_width/2, 455.-pixel_width/2, 455.+pixel_width/2, # PMT 9
             461.+pixel_width/2, 467.+pixel_width/2, 473.+pixel_width/2, 479.+pixel_width/2, 485.+pixel_width/2,
             491.+pixel_width/2, 497.+pixel_width/2, 505.-pixel_width/2, 505.+pixel_width/2, # PMT 10
             511.+pixel_width/2, 517.+pixel_width/2, 523.+pixel_width/2, 529.+pixel_width/2, 535.+pixel_width/2,
             541.+pixel_width/2, 547.+pixel_width/2, 555.-pixel_width/2, 555.+pixel_width/2, # PMT 11
             561.+pixel_width/2, 567.+pixel_width/2, 573.+pixel_width/2, 579.+pixel_width/2, 585.+pixel_width/2, 
             591.+pixel_width/2, 597.+pixel_width/2, 605.-pixel_width/2, 605.+pixel_width/2, # PMT 12
             611.+pixel_width/2, 617.+pixel_width/2, 623.+pixel_width/2, 629.+pixel_width/2, 635.+pixel_width/2,
             641.+pixel_width/2, 647.+pixel_width/2, 655.-pixel_width/2, 655.+pixel_width/2, # PMT 13
             661.+pixel_width/2, 667.+pixel_width/2, 673.+pixel_width/2, 679.+pixel_width/2, 685.+pixel_width/2,
             691.+pixel_width/2, 697.+pixel_width/2, 705.-pixel_width/2, 705.+pixel_width/2, # PMT 14
             711.+pixel_width/2, 717.+pixel_width/2, 723.+pixel_width/2, 729.+pixel_width/2, 735.+pixel_width/2,
             741.+pixel_width/2, 747.+pixel_width/2, 755.-pixel_width/2, 755.+pixel_width/2, # PMT 15
             761.+pixel_width/2, 767.+pixel_width/2, 773.+pixel_width/2, 779.+pixel_width/2, 785.+pixel_width/2,
             791.+pixel_width/2, 797.+pixel_width/2, 805.-pixel_width/2, 805.+pixel_width/2, # PMT 16
             811.+pixel_width/2, 817.+pixel_width/2, 823.+pixel_width/2, 829.+pixel_width/2, 835.+pixel_width/2,
             841.+pixel_width/2, 847.+pixel_width/2, 855.-pixel_width/2, 855.+pixel_width/2, # PMT 17
             861.+pixel_width/2, 867.+pixel_width/2, 873.+pixel_width/2, 879.+pixel_width/2, 885.+pixel_width/2,
             891.+pixel_width/2, 897.+pixel_width/2, 897.+pixel_width/2 + 2]


    
bins_y = [0,5.-pixel_width/2, 5. + pixel_width/2,  11.+pixel_width/2,  17.+pixel_width/2,  23.+pixel_width/2, 
             29.+pixel_width/2,  35+pixel_width/2.,  41.+pixel_width/2,  47.+pixel_width/2, 55. - pixel_width/2, 55. +pixel_width/2, # PMT 1 
             61.+pixel_width/2,  67.+pixel_width/2, 73.+pixel_width/2,  79.+pixel_width/2,  85.+pixel_width/2,
             91.+pixel_width/2,  97.+pixel_width/2, 105.-pixel_width/2, 105.+pixel_width/2, # PMT 2
             111.+pixel_width/2, 117.+pixel_width/2, 123.+pixel_width/2, 129.+pixel_width/2, 135.+pixel_width/2,
             141.+pixel_width/2, 147.+pixel_width/2, 155.-pixel_width/2, 155.+pixel_width/2, # PMT 3
             161.+pixel_width/2, 167.+pixel_width/2, 173.+pixel_width/2, 179.+pixel_width/2, 185.+pixel_width/2,
             191.+pixel_width/2, 197.+pixel_width/2, 205.-pixel_width/2,205.+pixel_width/2, # PMT 4
             211.+pixel_width/2, 217.+pixel_width/2, 223.+pixel_width/2, 229.+pixel_width/2, 235.+pixel_width/2,
             241.+pixel_width/2, 247.+pixel_width/2, 255.-pixel_width/2,255.+pixel_width/2, # PMT 5
             261.+pixel_width/2, 267.+pixel_width/2, 273.+pixel_width/2, 279.+pixel_width/2, 285.+pixel_width/2,
             291.+pixel_width/2, 297.+pixel_width/2, 297.+pixel_width/2 + 2] # PMT 6

# bins_y = [0, 2.0, 8.0, 8.0, 14.0, 14.0, 20.0, 20.0, 26.0, 26.0, 32.0, 32.0, 38.0, 38.0, 44.0, 44.0, 50.0, 52.0, 52.0, 58.0, 58.0, 64.0, 64.0, 70.0, 70.0, 76.0, 76.0, 82.0, 82.0, 88.0, 88.0, 94.0, 94.0, 100.0, 102.0, 102.0, 108.0, 108.0, 114.0, 114.0, 120.0, 120.0, 126.0, 126.0, 132.0, 132.0, 138.0, 138.0, 144.0, 144.0, 150.0, 152.0, 152.0, 158.0, 158.0, 164.0, 164.0, 170.0, 170.0, 176.0, 176.0, 182.0, 182.0, 188.0, 188.0, 194.0, 194.0, 200.0, 202.0, 202.0, 208.0, 208.0, 214.0, 214.0, 220.0, 220.0, 226.0, 226.0, 232.0, 232.0, 238.0, 238.0, 244.0, 244.0, 250.0, 252.0, 252.0, 258.0, 258.0, 264.0, 264.0, 270.0, 270.0, 276.0, 276.0, 282.0, 282.0, 288.0, 288.0, 294.0, 294.0, 300.0]

# bins_x = [0, 2.0, 8.0, 8.0, 14.0, 14.0, 20.0, 20.0, 26.0, 26.0, 32.0, 32.0, 38.0, 38.0, 44.0, 44.0, 50.0, 52.0, 58.0, 58.0, 64.0, 64.0, 70.0, 70.0, 76.0, 76.0, 82.0, 82.0, 88.0, 88.0, 94.0, 94.0, 100.0, 102.0, 108.0, 108.0, 114.0, 114.0, 120.0, 120.0, 126.0, 126.0, 132.0, 132.0, 138.0, 138.0, 144.0, 144.0, 150.0, 152.0, 158.0, 158.0, 164.0, 164.0, 170.0, 170.0, 176.0, 176.0, 182.0, 182.0, 188.0, 188.0, 194.0, 194.0, 200.0, 202.0, 208.0, 208.0, 214.0, 214.0, 220.0, 220.0, 226.0, 226.0, 232.0, 232.0, 238.0, 238.0, 244.0, 244.0, 250.0, 252.0, 258.0, 258.0, 264.0, 264.0, 270.0, 270.0, 276.0, 276.0, 282.0, 282.0, 288.0, 288.0, 294.0, 294.0, 300.0, 302.0, 308.0, 308.0, 314.0, 314.0, 320.0, 320.0, 326.0, 326.0, 332.0, 332.0, 338.0, 338.0, 344.0, 344.0, 350.0, 352.0, 358.0, 358.0, 364.0, 364.0, 370.0, 370.0, 376.0, 376.0, 382.0, 382.0, 388.0, 388.0, 394.0, 394.0, 400.0, 402.0, 408.0, 408.0, 414.0, 414.0, 420.0, 420.0, 426.0, 426.0, 432.0, 432.0, 438.0, 438.0, 444.0, 444.0, 450.0, 452.0, 458.0, 458.0, 464.0, 464.0, 470.0, 470.0, 476.0, 476.0, 482.0, 482.0, 488.0, 488.0, 494.0, 494.0, 500.0, 502.0, 508.0, 508.0, 514.0, 514.0, 520.0, 520.0, 526.0, 526.0, 532.0, 532.0, 538.0, 538.0, 544.0, 544.0, 550.0, 552.0, 558.0, 558.0, 564.0, 564.0, 570.0, 570.0, 576.0, 576.0, 582.0, 582.0, 588.0, 588.0, 594.0, 594.0, 600.0, 602.0, 608.0, 608.0, 614.0, 614.0, 620.0, 620.0, 626.0, 626.0, 632.0, 632.0, 638.0, 638.0, 644.0, 644.0, 650.0, 652.0, 658.0, 658.0, 664.0, 664.0, 670.0, 670.0, 676.0, 676.0, 682.0, 682.0, 688.0, 688.0, 694.0, 694.0, 700.0, 702.0, 708.0, 708.0, 714.0, 714.0, 720.0, 720.0, 726.0, 726.0, 732.0, 732.0, 738.0, 738.0, 744.0, 744.0, 750.0, 752.0, 758.0, 758.0, 764.0, 764.0, 770.0, 770.0, 776.0, 776.0, 782.0, 782.0, 788.0, 788.0, 794.0, 794.0, 800.0, 802.0, 808.0, 808.0, 814.0, 814.0, 820.0, 820.0, 826.0, 826.0, 832.0, 832.0, 838.0, 838.0, 844.0, 844.0, 850.0, 852.0, 858.0, 858.0, 864.0, 864.0, 870.0, 870.0, 876.0, 876.0, 882.0, 882.0, 888.0, 888.0, 894.0, 894.0, 900.0, 902.0]