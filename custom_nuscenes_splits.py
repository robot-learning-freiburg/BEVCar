# custom DAY/RAIN/NIGHT split of the nuscenes val split
# adapted from nuscenes/utils/splits.py

# mapping from scene_names to scene_tokens
"""
scene-0003: e7ef871f77f44331aefdebc24ec034b7
scene-0012: 265f002f02d447ad9074813292eef75e
scene-0013: 41fde20fedcd4d22ab26811688612870
scene-0014: d1e57234fd6a463d963670938f9f556e
scene-0015: 813213458a214a39a1d1fc77fa52fa34
scene-0016: efa5c96f05594f41a2498eb9f2e7ad99
scene-0017: 3dd9ad3f963e4f588d75c112cbf07f56
scene-0018: b51869782c0e464b8021eb798609f35f
scene-0035: 3dd2be428534403ba150a0b60abc6a0a
scene-0036: 5a0dd8908a3a459b83ec5eb6ac7d0f82
scene-0038: 2f56eb47c64f43df8902d9f88aa8a019
scene-0039: 8edbc31083ab4fb187626e5b3c0411f7
scene-0092: 7365495b74464629813b41eacdb711af
scene-0093: cba3ddd5c3664a43b6a08e586e094900
scene-0094: 91c071bcc1ad4fa1b555399e1cfbab79
scene-0095: b4b82c4d338a4b6d86835388ce076345
scene-0096: 68e79a88244f447f993a72da444b29ba
scene-0097: 2eb4d7f00e584a548aa0b899638bfb0a
scene-0098: c65c4acf86954f8cbd53a3541a3bfa3a
scene-0099: 5af9c7f124d84e7e9ac729fafa40ea01
scene-0100: afd73f70ff7d46d6b772d341c08e31a5
scene-0101: 01452fbfbf4543af8acdfd3e8a1ee806
scene-0102: ddb615d9bb22484cabc6545b632a1025
scene-0103: fcbccedd61424f1b85dcbf8f897f9754
scene-0104: c525507ee2ef4c6d8bb64b0e0cf0dd32
scene-0105: 2ed0fcbfc214478ca3b3ce013e7723ba
scene-0106: bed67ef03c4a4066a74f6c0117d512ee
scene-0107: 3b2ee26cb8484f77895bc336663df502
scene-0108: a178a1b5415f45c08d179bd2cacdf284
scene-0109: fb73d1a6c16147ee9416faf6b310fadb
scene-0110: d87ec0461e3f4c4ea29b0f0c0ede5167
scene-0221: acc29386502047339e1ec6b9c7e512d2
scene-0268: c3ab8ee2c1a54068a72d7eb4cf22e43d
scene-0269: 9f1f69646d644e35be4fe0122a8b91ef
scene-0270: 2ca15f59d656489a8b1a0be4d9bead4e
scene-0271: 8180a1dbbba3479bb0c7f4ff6e9a3f0e
scene-0272: 080a52cb8f59489b9cddc7b721808088
scene-0273: 1aa633b683174280b243e0a9a7ad9171
scene-0274: 9709626638f5406f9f773348150d02fd
scene-0275: 8377abf77f464a9cb62eacf63f383422
scene-0276: 201b7c65a61f4bc1a2333ea90ba9a932
scene-0277: 848ac962547c4508b8f3b0fcc8d53270
scene-0278: 91f797db8fb34ae5b32ba85eecae47c9
scene-0329: 5eaff323af5b4df4b864dbd433aeb75c
scene-0330: 3927699fa562451a98bb13bf8405361d
scene-0331: 9d1307e95c524ca4a51e03087bd57c29
scene-0332: 93608f0d57794ba6b014314c488e2b4a
scene-0344: 6d872d1448814fd189e631b1187c3771
scene-0345: fcc020250f884397965ba00c1d9ad9e6
scene-0346: a499ee875da34e2b9655afb999edb8a9
scene-0519: b07358651c604e2d83da7c4d4755de73
scene-0520: 26a6b03c8e2f4e6692f174a7074e54ff
scene-0521: 4962cb207a824e57bd10a2af49354b16
scene-0522: 2abb3f3517c64446a5768df5665da49d
scene-0523: d7bacba9119840f78f3c804134ceece0
scene-0524: 6a24a80e2ea3493c81f5fdb9fe78c28a
scene-0552: 16e50a63b809463099cb4c378fe0641e
scene-0553: 6f83169d067343658251f72e1dd17dbc
scene-0554: f2541977a00a448db9b46a32187d9059
scene-0555: 3363f396bb43405fbdd17d65dc123d4e
scene-0556: b94fbf78579f4ff5ab5dbd897d5e3199
scene-0557: 3f90afe9f7dc49399347ae1626502aed
scene-0558: 952cb0bcd89b4ca4b904cdcbbf595523
scene-0559: 0e7ede02718341558414865d5c604745
scene-0560: 44c9089913db4d4ab839a2fcb35989ed
scene-0561: ed242d80ccb34b139aaf9ab89859332e
scene-0562: 30a1a4ccd60047b4a22ee3bf0645f3ad
scene-0563: 8e3364691ee94c698458e0913a29af78
scene-0564: 84e056bd8e994362a37cba45c0f75558
scene-0565: 54f56f80350b4c07af598ee87cf3886a
scene-0770: aedcd3cf7c4a49d7a4a43ab7443a9eb1
scene-0775: 96e5f1f0944946f391b4ef33ad623008
scene-0777: 4bb9626f13184da7a67ba1b241c19558
scene-0778: ca6e45c25d954dc4af6e12dd7b23454d
scene-0780: 656bb27689dc4e9b8e4559e3f6a7e534
scene-0781: 57dc3221a3d845b5ab17ff0f98ce336f
scene-0782: 7061c08f7eec4495979a0cf68ab6bb79
scene-0783: 40209c4e465d4b4e8341ebd52be0d842
scene-0784: 50ff554b3ecb4d208849d042b7643715
scene-0794: 905cfed4f0fc46679e8df8890cca4141
scene-0795: 36f27b26ef4c423c9b79ac984dc33bae
scene-0796: c5224b9b454b4ded9b5d2d2634bbda8a
scene-0797: f24b4682fbb9482aba149534deed1cc9
scene-0798: 380ff00ec86447e3b986edc8e82ffba7
scene-0799: ec7b7459461e4da1a236ba23e22377c9
scene-0800: 07aed9dae37340a997535ad99138e243
scene-0802: 6741d407b1b44511853e5ec7aaee2992
scene-0916: 325cef682f064c55a255f2625c533b75
scene-0917: 223096a415cc45bf8ecd4c3a42251fd7
scene-0919: 76ceedbcc6a54b158eba9945a160a7bc
scene-0920: 955ff42a1990442d868cdfc0c01583d4
scene-0921: 112ca771e318478a88cfa692f61ffcac
scene-0922: 04219bfdc9004ba2af16d3079ecc4353
scene-0923: ee48ee50025e40b4ae27dbdf84d92825
scene-0924: 32185f91e68f4069ab3cdd2f4f1a4ff1
scene-0925: 696a45dbd11346b794fdce43fa0a1770
scene-0926: c4df079d260241ff8015218e29b42ea7
scene-0927: 5521cd85ed0e441f8d23938ed09099dd
scene-0928: e036014a715945aa965f4ec24e8639c9
scene-0929: cb3e964697d448b3bc04a9bc06c9e131
scene-0930: a04daf2d0f194b2ab2ff2a47dfebc1d7
scene-0931: e60ef590e3614187b7800db3e5284e1a
scene-0962: 2086743226764f268fe8d4b0b7c19590
scene-0963: 4efbf4c0b77f467385fc2e19da45c989
scene-0966: c92fdd793fbb4401b12782a9b8d4a499
scene-0967: 6776b91389394ff18e57b269863b4dbf
scene-0968: 931c5c57011944459bba3825ab8777a9
scene-0969: aacd6706a091407fb1b0e5343d27da7e
scene-0971: 64bfc5edd71147858ce7446892d7f864
scene-0972: e005041f659c47e194cd5b18ea6fc346
scene-0625: 0ac05652a4c44374998be876ba5cd6fd
scene-0626: 3ada261efee347cba2e7557794f1aec8
scene-0627: 5301151d8b6a42b0b252e95634bd3995
scene-0629: 991d65cab952449a821deb32e971ff19
scene-0630: 9068766ee9374872a380fe75fcfb299e
scene-0632: b789de07180846cc972118ee6d1fb027
scene-0633: 9088db17416043e5880a53178bfa461c
scene-0634: 7210f928860043b5a7e0d3dd4b3e80ff
scene-0635: 7bd098ac88cb4221addd19202a7ea5de
scene-0636: 19d97841d6f64eba9f6eb9b6e8c257dc
scene-0637: 3045ed93c2534ec2a5cabea89b186bd9
scene-0638: f5b29a1e09d04355adcd60ab72de006b
scene-0904: c164a8e8e8b8489f964f711f472789be
scene-0905: d3b86ca0a17840109e9e049b3dd40037
scene-0906: dce6f3f2bf6b4859abcf3268581969d3
scene-0907: 01e4fcbe6e49483293ce45727152b36e
scene-0908: d01e7279da2649ef896dc42f6b9ee7ab
scene-0909: 7e8ff24069ff4023ac699669b2c920de
scene-0910: a7d073bc435b4356a0a9a5ebfb61f229
scene-0911: 85651af9c04945c3a394cf845cb480a6
scene-0912: 197a7e4d3de84e57af17b3d65fcb3893
scene-0913: 55b3a17359014f398b6bbd90e94a8e1b
scene-0914: a2b005c4dd654af48194ada18662c8ca
scene-0915: 3a2d9bf6115f40898005d1c1df2b7282
scene-1059: 034dee1695304630b0692da8c1f153fc
scene-1060: 7052d21b95fc4bae8761b8d9524f3e42
scene-1061: e8099a6136804f3bb9b38ff94d98eb64
scene-1062: e6f1a7e6218a4737bfedc6af90926b3e
scene-1063: 6af9b75e439e4811ad3b04dc2220657a
scene-1064: f97bf749746c4c3a8ad9f1c11eab6444
scene-1065: bd338b912ce9434995b29b6dac9fbf1d
scene-1066: afbc2583cc324938b2e8931d42c83e6b
scene-1067: 16be583c31a2403caa6c158bb55ae616
scene-1068: e8834785d9ff4783a5950281a4579943
scene-1069: d29527ec841045d18d04a933e7a0afd2
scene-1070: 85889db3628342a482c5c0255e5347e9
scene-1071: 0dae482684ce4cd69a7258f55bc98d73
scene-1072: 82cabcf15a1a48aca6bbf082f8710e39
scene-1073: 6498fce2f38645fc9bf9d4464b159230
"""

val_day = \
        ['scene-0003', 'scene-0012', 'scene-0013', 'scene-0014', 'scene-0015', 'scene-0016', 'scene-0017', 'scene-0018',
         'scene-0035', 'scene-0036',
         'scene-0038', 'scene-0039', 'scene-0092', 'scene-0093', 'scene-0094', 'scene-0095', 'scene-0096', 'scene-0097',
         'scene-0098', 'scene-0099',
         'scene-0100', 'scene-0101', 'scene-0102', 'scene-0103', 'scene-0104', 'scene-0105', 'scene-0106', 'scene-0107',
         'scene-0108', 'scene-0109',
         'scene-0110', 'scene-0221', 'scene-0268', 'scene-0269', 'scene-0270', 'scene-0271', 'scene-0272', 'scene-0273',
         'scene-0274', 'scene-0275',
         'scene-0276', 'scene-0277', 'scene-0278', 'scene-0329', 'scene-0330', 'scene-0331', 'scene-0332', 'scene-0344',
         'scene-0345', 'scene-0346',
         'scene-0519', 'scene-0520', 'scene-0521', 'scene-0522', 'scene-0523', 'scene-0524', 'scene-0552', 'scene-0553',
         'scene-0554', 'scene-0555',
         'scene-0556', 'scene-0557', 'scene-0558', 'scene-0559', 'scene-0560', 'scene-0561', 'scene-0562', 'scene-0563',
         'scene-0564', 'scene-0565',
         'scene-0770', 'scene-0771', 'scene-0775', 'scene-0777', 'scene-0778', 'scene-0780', 'scene-0781', 'scene-0782',
         'scene-0783', 'scene-0784',
         'scene-0794', 'scene-0795', 'scene-0796', 'scene-0797', 'scene-0798', 'scene-0799', 'scene-0800', 'scene-0802',
         'scene-0916', 'scene-0917',
         'scene-0919', 'scene-0920', 'scene-0921', 'scene-0922', 'scene-0923', 'scene-0924', 'scene-0925', 'scene-0926',
         'scene-0927', 'scene-0928',
         'scene-0929', 'scene-0930', 'scene-0931', 'scene-0962', 'scene-0963', 'scene-0966', 'scene-0967', 'scene-0968',
         'scene-0969', 'scene-0971',
         'scene-0972']

val_rain = \
         ['scene-0625', 'scene-0626', 'scene-0627', 'scene-0629', 'scene-0630', 'scene-0632', 'scene-0633',
          'scene-0634', 'scene-0635', 'scene-0636',
          'scene-0637', 'scene-0638', 'scene-0904', 'scene-0905', 'scene-0906', 'scene-0907', 'scene-0908',
          'scene-0909', 'scene-0910', 'scene-0911',
          'scene-0912', 'scene-0913', 'scene-0914', 'scene-0915']

val_night = \
          ['scene-1059', 'scene-1060', 'scene-1061', 'scene-1062', 'scene-1063', 'scene-1064', 'scene-1065',
           'scene-1066', 'scene-1067', 'scene-1068',
           'scene-1069', 'scene-1070', 'scene-1071', 'scene-1072', 'scene-1073']


def create_drn_eval_split_scenes(verbose: bool = False) -> dict[str, list[str]]:
    """
    Similar to create_splits_scenes() in nuscenes, but returns a mapping from split to scene names, for
    day, rain, night (drn) or all scenes in that new order.
    The splits are as follows:
    - val_day: all day scenes without rain
    - val_rain: all rainy scenes (day and night)
    - val_night: all night scenes
    - val_all: val_day + val_rain + val_night

    Args:
        verbose (bool): Whether to print out statistics on a scene level.

    Returns:
        A mapping from split name to a list of scenes names in that split.
    """
    # Use hard-coded splits.
    val_all = val_day + val_rain + val_night
    assert len(val_all) == 150 and len(set(val_all)) == 150, 'Error: Splits incomplete!'

    scene_splits = {'val_day': val_day,         # idx: 0 to 110     len: 111
                    'val_rain': val_rain,       # idx: 111 to 134   len:  24
                    'val_night': val_night,     # idx: 135 to 149   len:  15
                    'val_all': val_all
                    }

    # Optional: Print scene-level stats.
    if verbose:
        for split, scenes in scene_splits.items():
            print('%s: %d' % (split, len(scenes)))
            print('%s' % scenes)

    return scene_splits
