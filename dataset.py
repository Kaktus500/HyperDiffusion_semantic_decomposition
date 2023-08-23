import os

import numpy as np
import torch

from augment import random_permute_flat, random_permute_mlp, sorted_permute_mlp
from math import floor, ceil
import trimesh
from trimesh.voxel import creation as vox_creation
from torch.utils.data import Dataset
from os.path import join

from hd_utils import get_mlp, generate_mlp_from_weights
from siren.dataio import anime_read


class VoxelDataset(Dataset):
    def __init__(self, mesh_folder, wandb_logger, model_dims, mlp_kwargs, cfg, object_names=None):
        self.mesh_folder = mesh_folder
        if len(cfg.filter_bad_path) == 0:
            blacklist = {'f3e2df468c15795872517bb0a6b4d3ef', '51ebcde47b4c29d81a62197a72f89474',
                         '46c259e87609c54fafc0cc47720c0ef4', 'a2b758aa5d51642bd32761b337f8b72a',
                         '8ddc3bfd5a86d4be2e7c68eb5d1b9123', 'd6d84d05892a4f492e7c68eb5d1b9123',
                         'dbdca81a0f9079096d511e9563e4bbe7', '3badd7765618bd66a532a2f6f060af39',
                         '3fa5f65a92a5e521d87c63d8b3018b58', '4bfa5d948d9ca7ab7c5f0032facde6fe',
                         '46d2373fa05a4876bc913a3935f0ae10', 'bc16d2cf6420432cb87697d3904b168b',
                         'ce4b8076f8f9a5a05be07e24c1d3227d', 'a61a59a4c48154db37678474be485ca',
                         'daedff5e78136a8b507c9a5cb0f72e1e', 'd2e2e23f5be557e2d1ab3b031c100cb1',
                         '9c7268dd8ec3a3bb590874dcd9dc8481', 'd54a694d514b1911844ac48bcfce34',
                         '29f514bdcf72e779bbf3143b1cb6076a', '5392580fe70da9043554ccf8c30febe7',
                         'c47a2cb459ab849c493ab0b98cb45d3', '67a6b5b12ba64c529a6e43b878d5b335',
                         '947a78a898b6aa81f19a675dcc5ca632', '413a85d9cc7f19a8b6c3e7b944b34fa',
                         '6778c46aff633538c0676369cd1d063d', 'd2bf5f39131584c0a8cba409c4409ba9',
                         '337658edebb67c301ce9f50324082ee4', '5a6eb0f2a316f23666cf1b4a8fc3914e',
                         '2c932237239e4d22181acc12f598af7', '72aedca98ea4429c8ed4db287040dac1',
                         'bc92b144ec7029782e7c68eb5d1b9123', '75705b82eb24f5ae23e79e99b949a341',
                         '2d4a57a467307d675e9e2656aff7dd5b', 'c61f67c3f03b10537f3efc94c2d31dc5',
                         '1c27d282735f81211063b9885ddcbb1', 'a20db17555286e06f5e83e93ffcfd3f0',
                         '33b8b6af08696fdea616caf97d73fa02', 'dba3ab64bab4ed3ed13ef00338ba8c52',
                         '65d7ed8984d78a797c9af13aaa662e8e', '83b55a6e1677563d6ae6891f58c50f',
                         '9f90f3298d7b6c6f938204d34a891739', '71f718d82a865472bfa44fe936def6d4',
                         '65166f18d4ee36a61cac9affde84fd21', '8f4416cb67c3807bcf155ddc002a8f77',
                         'ec879d7360572a5db65bd29086fc2a67', 'c107063f573cbfb3ca8607f540cc62ba',
                         '3c52f7f687ce7efe3df325aac2f73830', 'e06d3e6c1fb4b208cb7c15fd62c3982e',
                         '775120d01da7a7cc666b8bccf7d1f46a', '523f1c2392338cf2b7f9c6f6b7bee458',
                         '6e324581dbbdfb5548e8405d6c51a908', 'fc7387d630c84bb9c863ab010b80d9ed',
                         '3a82056ea319a442f64801ad2940cdd5', '508fa09e7df177e7fee8021c81c84603',
                         '72c28618e3273795f9066cd40fcf015', '5f2aa0bc5a7fc2beacf77e718d93f3e1',
                         '22ce6f6d1adb08d7bbf3143b1cb6076a', '25bd1569261bc545e8323edc0fe816a8',
                         '723c87de224355b69878ac4a791083c5', '4937396b74bc16eaf781741e31f0df4',
                         '65b3e612c00a65a6bc0909d98a1ff2b4', 'b8e4e4994d4674cf2023ec956848b741',
                         '6b84749eaca0e657f37f38dedb2f1219', '220a911e2e303865f64801ad2940cdd5',
                         'cc113b6e9d4fbeb23df325aac2f73830', 'f8ceed6c984895079a6e43b878d5b335',
                         '2ba980d080f89581ab2a0ebad7754fba', 'f9db62e6a88f0d7129343faf3bbffb15',
                         '6d0b546bb6256eb5e66cabd11ba41eae', '794fe891e20800d37bbc9fbc6d1fd31d',
                         '6615bb23e68159c193d4024985440d4c', '28402efe2e3435d124fbfb66ea1f14f7',
                         'a4f5529edc990706e719b5362fe06bbb', '35835c28f33d985fa18e0889038e4fb6',
                         'fbf6917bdd86d5862df404314e891e08', '1d269dbde96f067966cf1b4a8fc3914e',
                         '460cf3a75d8467d1bb579d1d8d989550', '35131f7ea7256373879c08e5cc6e64bc',
                         'd581b2e10f8bc6aecc1d733a19631a1', 'fb62efc64c58d1e5e0d07a8ce78b9182',
                         '580e54df8765aac1c1ca96a73599ca7e', '2aec6e6096e640add00d52e62bf14ee9',
                         '4d50ff789e84e70e54eefcdc602d4520', 'c1b5fd196374203d772a4b403daac29b',
                         '166d333d38897d1513d521050081b441', '3360b510b0408682bbf3143b1cb6076a',
                         'f390b1b28b6dda03dc57b3e43c28d486', '4bd5f77521e76e6a2e690fa6dfd5d610',
                         'f25da5eca572f01bd8d90babf3c5509a', '796bb7d6f4d5ce8471d03b466c72ce41',
                         '460f2b6d8f4dc18d565895440030d853', '7488f87b1e08bbabd00d52e62bf14ee9',
                         'f6f5efa1554038ce2154ead1f7ab03aa', '6dccca8485094190be13ce34aa7c0c1c',
                         'e521828113e1e0c45e28caa3b26a73fd', 'e24be0487f6a3a92d4fd6e0556ecdd3d',
                         '494660cc290492218ac43fbf276bac06', '350d12f5290908c7f446f92b52bbd82a',
                         'ccb9e8dd6453b6c2a2981e27422ad4a7', '7805239ad1e40e07c69d7040c52664c5',
                         'ce2c4502b06c3c356abde8e1529c422f', '53011f6c40df3d7b4f95630cc18536e0',
                         '1b626fd06226b600adcbeb54f3d014e9', '8f33d2c0c08befe48caa71b1fbf7fb98',
                         'dcb5bded8772135b4295343ee7255799', '6a75658fb8242b9c590874dcd9dc8481',
                         '89f21747e6403735d9e1375411b582e', '1df217928b39b927a8cba409c4409ba9',
                         '3390c5050cda83c09a6e43b878d5b335', 'e66692f2ec8ea898874f1daffc45b57c',
                         '934e5dbb193dab924d02f09e7a1a5b28', '3844797c89c5e2e821b85e5214b0d6a7',
                         'd2f8a99bbdc387c8c5552bebbfa48bd7', 'e25e3dc95243a92c59bcb260e54d3785',
                         'd4a8134b2463a8bca8607f540cc62ba', 'c8db76f18f56c1344c2c88971423d0be',
                         '6420a3ff5e526d59e16519c843f95ce0', 'e8bd07a978ac82baef40e4c1c2686cd3',
                         'bf78d2e27a671fce4d4cb1c2a5e48b7a', '7a794db8180858fe90916c8815b5c43',
                         '9e524a14078824b5cfe15db7c5db913', 'e17a696c47d4292393db03f6b4e68f17',
                         'b0b164107a27986961f6e1cef6b8e434', '2f4f38d774e0e941dcc75fd1795fa3a5',
                         '2d255ec7a05b301a203936772104a82d', 'befcb95d80e0e49119ba010ddb4974fe',
                         'be96646f774c7d0e2a23d93958262ccc', '694da337c806f41a7209009cfb89d4bd',
                         '6058d6701a0ca4e748e8405d6c51a908', '6481eb70eb9a58cfb2bb87688c5d2f58',
                         '6fe837570383eb98f72a00ecdc268a5b', '17ad3ab1b1d12a7db26dc8ec64d633df',
                         '95a6c003f5bd8a4acef4e20384a35136', 'aa2af754642256c08699933784576e73',
                         '752a0bb6676c05bbe55e3ad998a1ecb4', '427030abcc0f11a8947bbeb9022263b8',
                         'd56fba80d39bdff738decdbba236bc1d', '3adbafd59a34d393eccd82bb51193a7f',
                         '91bd6e91455f85fddcf9f917545742df', 'bdc5360ff3c62ed69aa9d7f676c1fd7e',
                         '5c72cc06d79edd1cbbf3143b1cb6076a', '94cb05b56b89aa2223895f8c849f85e',
                         '4374a3b4b98e247b398db3ebdf468ed7', 'c58e9f5593a26f75e607cb6cb5eb37ba',
                         'b51032670accbd25d11b9a498c2f9ed5', 'edd9f45c7c927032db5e5b1ef1010d8b',
                         '6ea21a2265075beb9a2f7b9a6f4f875f', '82cd0676627c20b0879eac31511e27a8',
                         '265e6c77443c74bd8043fd2260891a82', 'd583d6f23c590f3ec672ad25c77a396',
                         'd63bd140efa537dcf73e5bc170b7e2be', 'f9e80ce23d9536623fddedb0bf24c68a',
                         '2c5bdd9a08122f9e2023ec956848b741', '4b20c1736440ff9d90dd6eb126f6cbbb',
                         '194098fb0aefc4a0666bd6da67d3abc2', 'be11ce096084bd784f95630cc18536e0',
                         'a61d4ad108af6eecca8607f540cc62ba', '2cc1ff07bcb27de4f64801ad2940cdd5',
                         'f96426f2392abb1d8d58389384d9812e', '395afa94dd4d549670e6bd9d4e2b211f',
                         'ffdaed95c3063cdf3ce4891c7dcdfb1c', '8bfcf5618f2d169c9a6e43b878d5b335',
                         '94d3a666d2dbc4385ff3feb917a6004b', 'c5e04f903e3472c31883411175051361',
                         'a9cdbca070047fe61e9dc95dd6a9f6', '9dbc0aba1311e7b8663e90eaf6b4ca52',
                         'a762fe69269fc34b2625832ae05a7344', '157a81baeb10914566cf1b4a8fc3914e',
                         'f7110ecac70994a83820d8f180caa23a', 'cbb5ed444249f0a9e90916c8815b5c43',
                         'dc7c5d12854b9467b96212c8f6cd06e', 'ba37c8ef604b675be1873a3963e0d14',
                         'a591b49406ab0d2abbf3143b1cb6076a', 'ebe0d0bfa6ec36edd88eab18f1be033b',
                         '9617ea33f17993e76f0c490da56d34b', '9baf5b31d70e0d05e98d814cc4d9c5e3',
                         '7e75688f4b185d4193a78ffd70737098', 'bee504b4909890df1dfabee9ba27dc70',
                         'c6bcec892954942a83855ba2afe73b0b', '2c9797204c91e3a440975e4feec771f6',
                         'a60b2775f1955153ca8607f540cc62ba', '552871abb864d06d35fc197bbabcd5bd',
                         'ce337df2f75801eeb07412c80bd835', '5aeb583ee6e0e4ea42d0e83abdfab1fd',
                         '880715a3ef69f47e62b600da24e0965', '873f4d2e92681d12709eb7790ef48e0c',
                         '8ec085a86e6d9425f4fb6842b3610149', '2a06adfb446c85c9f9d3f977c7090b2a',
                         'afd43430bd7c689f251fe573763aebef', '1eb3af39d93c8bf2ddea2f4a0e0f0d2e',
                         'eae54f8d47cf07c0aeec93871310e650', 'd75a4d4d25863f5062747c704d78d4f8',
                         '4e4128a2d12c818e5f38952c9fdf4604', 'af6b292c7857c78abb0e5c1799dab683',
                         'b87185699318f4b635fc197bbabcd5bd', 'bd22bcf18a9c8163adecb6fc00604132',
                         '70ed0fe305145f60e53236e6f2fbeb94', '4f1fb7c062c50fb15a2c5766752aea65',
                         '4f0bf26c62bb7c8b7e1c97634acf0214', '17e66cd463ff0de126360e1e29a956c7',
                         'c31f5303997022c842c20743f866e1a6', '3265b621ca222d29d00d52e62bf14ee9',
                         '41aafedd84a6fa7490baeef8ba5b93e5', '77ab8bb69221b13bbc0909d98a1ff2b4',
                         '802cbaaf1a51cf38c863ab010b80d9ed', 'e25794343ee37d6fa8eeb11153b68d81',
                         '62ebe2dd7bceccd097f28f82dd9c77a2', 'ade3c4987f49895ff960bc420d751255',
                         '755b0ee19aa7037453e01eb381ca65', '4e2322d4f1c0d29df96e777471c18dbe',
                         '1560968d05cd8887cc14f1e6f4f4f49b', 'c9584d90a1da19f723a665a253ac8cae',
                         'c85e3f6c572581de7d3b11085e75c7ad', '10aa040f470500c6a66ef8df4909ded9',
                         'd532a5abb756ebebcc14f1e6f4f4f49b', 'cf4c2d7d836c781e5a59e179e114b9a7',
                         'a1b95d2e1264f6bd66ccbd11a6ef0f68', 'e161df613fc808b0d7ec54df5db5828c',
                         '530d0edd6e2881653023dc1d1218bb2d', 'b3fbc7a0b0e3a821fd279055f27928f7',
                         'cc9b7118034278fcb4cdad9a5bf52dd5', 'cb8fb401a278fc36bbf3143b1cb6076a',
                         'c2d5bd1215248f9c8b6c29bda2bc905a', '9436273fc1a5e3ca7af159eaf7625abf',
                         'b5cdecafe353e18ac1006ed55bc1a3fc', '121b5c1c81aa77906b153e6e0582b3ac',
                         'f50eba69a3be1a1e536cfc00d8c31ac5', '7e8be9c88e596b84198609c994ea801',
                         'a051219f3f444fadc5e2bf7b5a5f1c56', '69d7ae2143c2a2709c6df67fce5fa25a',
                         '74ebf601d9c872a7828204947d78b9af', '92a83ecaa10e8d3f78e919a72d9a39e7',
                         'f5a8cae96024e709c1ccec171a275967', '8baeb2c664b0bdf4ca8607f540cc62ba',
                         'd3dcf83f03c7ad2bbc0909d98a1ff2b4', '1280f994ba1f92d28699933784576e73',
                         'a849d4325008fbea85dfb1711fe4ff6d', 'fc0dbd045c9391ce4a29fef5d2b2f3d7',
                         'b943b632fd36f75ac1ccec171a275967', 'e9df546c1e54af0733d72ae4e2c33411',
                         '57f1dfb095bbe82cafc7bdb2f8d1ea84', '1d0c128592385580e2129f6359ec27e3',
                         '844d36a369cdeed3ac4f72bf08dc79a6', 'dd48aa92170bdde04c3a35cee92bb95b',
                         'fb402a36e91ab1b44e7761521d3c6953', '1890f6391df25286394b1e418d5c594',
                         'fbb2e9c15888afcaca504cdc40c452de', '22944fabf8a9763f28132d87f74ffe13',
                         '71bb720c33da689090b1d6deb98feec6', 'efc7d4fb87937413dc13452e3008005b',
                         '389dd6f9d6730b0e29143caa6b05e24f', '13370c42b8ea45cf5e8172e6d9ae84ff',
                         '812111e3a4a545cbc863ab010b80d9ed', '1dbcb49dfbfd0844a480511cbe2c4655',
                         '150cdc45dabde04f7f29c61065b4dc5a', 'dfa36bffe436a98ee0534173b9189765',
                         '1d4ff34cdf90d6f9aa2d78d1b8d0b45c', 'e2275ee8d6b175f2f446f92b52bbd82a',
                         '1f672d2fd5e3f4e78026abe712c1ab05', 'fb0f2907b977e7cb67c5e3041553656b',
                         '6da4668de7ccdd0d4d10a13d437fced6', '66cd9502b63e8a97bbf3143b1cb6076a',
                         '7d180493022c01daace5721ccacba16', 'b2bb5a56b3d805b298b8c800ae001b66',
                         '828176e6eaee542ceb532c8487004b3c', '83dd9dd45724d5fbbeb310a83b693887',
                         'ed4aaf81dc577bedac4f72bf08dc79a6', 'eb7bf553e500b9b544bf3710e93f8cf7',
                         '3f69370401d4dc9a275386e1d3ac388e', '81596cc38eef8260ce9e5ac45c67ec22',
                         'ab35aa631852d30685dfb1711fe4ff6d', '9300dc1ca5f16b074f95630cc18536e0',
                         '6ba7cad8fa7301f9c1ca96a73599ca7e', 'fbd800d43c5f0d74250cb4f7fcd9ec03',
                         '7e1d4583f70c8583431589819e5ca60c', '8448475504ba9fdeca8607f540cc62ba',
                         'bfd606459cace196e7ee2e25a3cfaa4d', 'faa361f642620bb72def37e9c0b35d0e',
                         'e4ac77fdf28116232fa725d7a62a02a', '19604020a86ab1790b1d6deb98feec6',
                         '40192d0e50b4d2c1f27a705edb2f9ba6', '5f46f3c62e353c7bb4f5fdc59ce06e88',
                         '35e2eceef33804d8196c5820729d438f', 'e4e1b542f8732ae1c6768d9a3f27965',
                         'a2c5e769f19c7e97b7d7aa9b9ebcccb0', '5b048655453b37467584cbfee85fb982',
                         '95e589163afd0a7a609e2d916fa0da27', 'ec8ba88cdead53f336dafa9b6763ef3f',
                         'cf71f5442c4120db37678474be485ca', '3cd1b98f12d2a22bf3ad4b0977199f23',
                         '6dedeb5b87ee318b2154ead1f7ab03aa', '7442ad61d59fac8c7372dc0a2f1067b1',
                         '85396c57cdbf919f83467b3671ddaea2', 'c1260a89eee28413f2acf00738ce9d0d',
                         '189f045faacc1b5f9a8993cdad554625', 'eaa0d465e9d0c16acfbf0f1430c86945',
                         '6ce399fe42d54815e4406b3bf37e6ffe', 'e841e17e3256acf38699933784576e73',
                         'f6373cc88634e8ddaf781741e31f0df4', '36dd57178402cdf2afd477f714c68df9',
                         '8a674703723db7a390baeef8ba5b93e5', '6b6cb0c71731aacc277d303e3a640f98',
                         'e523ba4e79a48d31bd46d022fd7d80aa', 'de1a7b4e9911e48b48e8405d6c51a908',
                         '2cc44e0f9fb1efdb85e0a2007a11e92f', '6509073d1ff37d683d41f76be7f2e91f',
                         '9c7395d87c59aa54a79f2ed56427c6e6', 'a3e15e215144dda0a03ebab0e8b8f7a0',
                         '8f40518bd30467151e5ae32cb9e3711f', '70e4200e848e653072ec6e905035e5d7',
                         'f89b085c37779a5997517c313146c4ab', '92b7d0035cefb816d13ef00338ba8c52',
                         '795e0051d9ce7dfe384d4ad42dbd0045', '87fb26b8e56d25f2b87697d3904b168b',
                         '17c86b46990b54b65578b8865797aa0', '84b396dde837c81994445a3e8b9de59d',
                         'd16405b7a4300014ef5bed0139d3780c', 'ddfbeb997ef83cab884a857d19f7439f',
                         '4fccf49d6356c756b833a96759a020e2', 'b31bbc50a0d3a4366cf1b4a8fc3914e',
                         'fc5dade8622f686b4aba1f0cb15b1439', '30b514b24624da4fc1ccec171a275967',
                         '25a057e935aeb4b6842007370970c479', '519f1ddcbf942a76a71b0c9b506dc672',
                         'f59a2be8fd084418bbf3143b1cb6076a', '5cbe5be753b5c7faf389d19fad321c37',
                         '5ae05c956af61890b58b3ab5dbaea0f7', '4eced94670d10b35e856faf938562bd0',
                         '396312e9bec88c2590b1d6deb98feec6', 'd3f93b6da62fae46a98ae8c73b190485',
                         '8eeb9f69fc9ef1b0b45fd154bd3b6957', 'af69c8020fa9b68366cf1b4a8fc3914e',
                         '4c880eae29fd97c1f9575f483c69ee5', '909f59399d056983a0a3307f8f7721fc',
                         'd45772f938d14d52736e55e9ba083986', 'f7c11b5e07e9ccab3a116bc3efac4e3b',
                         '57a57f639a3e636d914c075742032f6', '96600318559071d48caa71b1fbf7fb98',
                         'fcd7a8834a7f26f15069db69b8c1c70', 'eed299b690be51ffbd931fcaa69140',
                         '6cf339eb8c950ac5d556fc099c90ab45', 'aeb538b2f1c36a8d9e811b082458229e',
                         '29120728132e5dce42a2048a31b7df8c', '4e4ae13865bf47f41adbb4c4e06ad649',
                         'ecbb6df185a7b260760d31bf9510e4b7', 'afd02e6d4cf0a342c1ccec171a275967',
                         '751b1e75fcd7f1deffb814dfce3ab22e', 'fbebcde2d8fbf81ee7cf320ab5761d45',
                         '757c47e20a37647431e38f024b7ad042', 'a87adc4fb1229b7f6d0f2528dd6f85b9',
                         'ad6e93a1db3e1da5977e4bb19a62128e', '393cfa7e090b972dce2cec85ea6ae00d',
                         'c950fc7d559f30016e86a8ae6e1f4d7e', 'd390f0246fd43cc8bd46d022fd7d80aa',
                         'cb1aff81a3c6ef71d25fd5e3115979a5', '381111f176565d48fe4c91be246ef13b',
                         'c3733e626c07b9ff26360e1e29a956c7', '800334df5da57266a4642ec4b6f68a',
                         'b9e6298004d7d422bd46d022fd7d80aa', 'f8647af0b1ad01445de04aad18bd94c3',
                         'd0001af4b3c1fe3d6f675e9f2e677792', 'db73a3c857949f469a6e43b878d5b335',
                         '62fe06fd4f1b390fa9bcc7eaa4032fa4', '959044f10e27b89ee664ce1de3ddc8b4',
                         'e58010dd5766e0ce78f081615c34707c', '299ec43108d7109113ae47e860a2333a',
                         'f562ff06e51e573e42979ff355194f16', 'abbe69a6f94918c79eb9aa3111a82815',
                         '37f2f187a1582704a29fef5d2b2f3d7', 'a3c928995562fca8ca8607f540cc62ba',
                         '7f895411162624e92023ec956848b741', 'c0c32558decf271df3ad4b0977199f23',
                         'ebedcd06f1770cd4bbf3143b1cb6076a', 'b702e35f4a59e81f64801ad2940cdd5',
                         'ee92bc35ee989f59271b3fb2659dec56', '8b72934186e1d8b0f510cd52a5f27547',
                         'ed0a9a32a8e35f21ca8607f540cc62ba', '48b99ae8fbe0762a8ed04761eced33c6',
                         '697b269a890237fe15796a932d10290d', 'f80343ac3064e74862347b4731688b0f',
                         '2893dc61be63a8a16d0ff49003c479bc', 'db4079b8c7d3d674ca8607f540cc62ba',
                         'd30689ca6cdf2601f551b6c3f174499e', 'd940f33afc01ff036da97d9f744f7e97',
                         '157bb84c08754307dff9b4d1071b12d7', 'e5a7a353d5fa8df844b2fa2cac0778f5',
                         '9483e1b0e4222cb4f2b0736dd4d8afe0', '965d457720def9a490b1d6deb98feec6',
                         'dd9a7dd5e2ea3389938204d34a891739', 'a2041f74f316e7b9585e3fa372e910b7',
                         'a361d82b2c510ca5208842e3d616cb23', '68c61d42222863682296d88107d065f6',
                         '3ac64a2c67cb44f19777d69c8d47140', 'e88e090caa1ccc5d187bd96066d7269e',
                         '4bb41171f7e6505bc32f927674bfca67', '12a1ac26d29ed1083554ccf8c30febe7',
                         '21827b0be78dd3e17dd9ca9e282b9209', '7175100f99a61c9646322bce65ca3756',
                         'b7b743834a6d78c2225a23c790f08fdd', '6ba642ca477a73db4c3a35cee92bb95b',
                         '2026699e25ba56c5fd6b49391fda17', '3f9cab3630160be9f19e1980c1653b79',
                         '4982bea0a007c19593b2f224b3acb952', '1ce5b9a79420e946bff7790df3158906',
                         'd172705764e25e20884a857d19f7439f', '52712e1c07ea494419ba010ddb4974fe',
                         'bc58ff3369054fa68f52dc705c3109b9', 'ed57671fc0252e15b95e9a91dc6bad16',
                         '7f837b389e885af471b4c018296f73c7', '1286826ff37699a1a0d713eb26379316',
                         'c0f9c28c45e7c8354f95630cc18536e0', 'd43b80dd95a2233a5ae839ffe09b9d31',
                         '22c11b2bab2cf93fc1ccec171a275967', '53edcc6832e776dcca8607f540cc62ba',
                         'ce682d7a2bbf77b6fc4b92d3d335214a', '839a950d0264cbb89a162c818d22a620',
                         '19a624cf1037fc75cda1835f53ae7d53', 'e9bae38bc2083d0bb4d73e4449062b04',
                         'f144e93fe2a11c1f4c3a35cee92bb95b', 'd109c08886c2a3dabdf566b587d6b21',
                         'e9f39176973edd33a8cba409c4409ba9', 'fbe788465e564e46bc0909d98a1ff2b4',
                         '85da8ecc055fc6cb58328b65a4733701', 'cbf4dc1c144ce656ffa79951b9f955a3',
                         '72a74e13c2424c19f2b0736dd4d8afe0', 'a287dc5d0e28d3d3325212819caa597d',
                         'b04ec55f4960b3b984b7ea000aa0a2b', '1a963a929d9b1332290d63dca780cfb6',
                         'b82731071bd39b66e4c15ad8a2edd2e', 'f13827d156628467b4cdad9a5bf52dd5',
                         '369244d49f8f1308b858e64ff0fa8db3', '14cd2f1de7f68bf3ab550998f901c8e1',
                         '7ee59463dc17ac6e3e3f3c9608255377', 'd80afa36aeb72c552b5147716975ed8a',
                         '98168c1772b769c0ea1bd6f2443b43e7', 'e452189bb7bd6617ef7cbef6334698fc',
                         '2b96f4b4896962473eb731541f9f8d', 'cb7c32bd7266daef37f38dedb2f1219',
                         '253a1aead30731904c3a35cee92bb95b', '8eda6d2dcf9302d2d041917556492646',
                         '6ad89740605331aef5f09964a6a1f97', 'e559a626d0ef8b4f982014dd9aabdeeb',
                         'da67955425ffe66071d03b466c72ce41', 'b80bd34ab330babbc8727b27ee96a4b7',
                         '95cfdf728da16975c5f6fdebb053ab2f', '78bd38a7282a73f8b184ba15dd506a2d',
                         'd78a16856adad344670aaa01f77ae41a', '947d6b9cd1966e2e719b5362fe06bbb',
                         '4c9214d70e0a00c6c1ccec171a275967', '5fed73635306ad9f14ac58bc87dcf2c2',
                         'e8c1e738997275799de8e648621673e1', '5f9b4ffc555c9915a3451bc89763f63c',
                         '5a37bc42a52130a18f52dc705c3109b9', 'c3408a7be501f09070d98a97e17b4da3',
                         '1f08b579e153b2de313f9af5275b7c70', 'ea58a51483e3604897dec65c2238cb8a',
                         '446f9144536c0e47f0c7ca80512a3ddf', '464a8718f0e81ffd9a6e43b878d5b335',
                         '2e3c317357ecb038543941eaaf04581f', 'd38922599bc74f6da30fd8ce49679098',
                         'e0a8ae255ed47518a847e990b54bf80c', '7addd02b1c255edcc863ab010b80d9ed',
                         '2628b6cfcf1a53465569af4484881d20', '85d3691b7bde76548b96ae1a0a8b84ec',
                         'a4ea22087dec2f32c7575c9089791ff', 'e954dc13308e6756308fc4195afc19d3',
                         'cc40acee83422fe892b90699bc4724f9', '4e67529b0ca7bd4fb3f2b01de37c0b29',
                         '34e87dd1c4922f7d48a263e43962eb7', '92e445da194d65873dc5bf61ec5f5588',
                         '64211a5d22e8ffad7209009cfb89d4bd', 'aaefbfb4765df684cf9f662004cc77d8',
                         '414f3305033ad38934f276985b6d695', 'c854bf983f2404bc15d4d2fdca08573e',
                         '2a3d485b0214d6a182389daa2190d234', '117830993cc5887726587cb13c78fb9b',
                         'a2491ac51414429e422ceeb181af6a7f', 'bcaf04bfae3afc1f4d48ad32fb72c8ce',
                         '8a84a26158da1db7668586dcfb752ad', '556363167281c6e486ecff2582325794',
                         '6d752b942618d6e38b424343280aeccb', '8b59ed9a391c86cdb4910ab5b756f3ae',
                         'd24f2a1da14a00ce16b34c3751bc447d', 'b5d0ae4f723bce81f119374ee5d5f944',
                         '542a1e7f0009339aa813ec663952445c', 'ae8a5344a37b2649eda3a29d4e1368cb',
                         'e2a6bed8b8920586c7a2c209f9742f15', '237b5aa80b3d3461d1d47c38683a697d'}

        else:
            blacklist = set(np.genfromtxt(cfg.filter_bad_path, dtype=str))

        self.mesh_files = []
        if object_names is None:
            self.mesh_files = [file for file in list(os.listdir(mesh_folder)) if
                               file not in ["train_split.lst", "test_split.lst", "val_split.lst"]]
        else:
            for file in list(os.listdir(mesh_folder)):
                if file.split(".")[0] in blacklist and cfg.filter_bad:
                    continue

                if ("_" in file and file.split("_")[1] in object_names) or file in object_names or file.split(".")[
                    0] in object_names:
                    self.mesh_files.append(file)
        self.transform = None
        self.logger = wandb_logger
        self.model_dims = model_dims
        self.cfg = cfg
        self.vox_folder = self.mesh_folder + "_vox"
        os.makedirs(self.vox_folder, exist_ok=True)

    def __getitem__(self, index):
        dir = self.mesh_files[index]
        path = join(self.mesh_folder, dir)
        resolution = self.cfg.vox_resolution
        voxel_size = 1.9 / (resolution - 1)
        total_time = self.cfg.unet_config.params.image_size
        if self.cfg.mlp_config.params.move:
            folder_name = os.path.basename(path)
            anime_file_path = os.path.join(path, folder_name + ".anime")
            nf, nv, nt, vert_data, face_data, offset_data = anime_read(anime_file_path)

            def normalize(obj, v_min, v_max):
                vertices = obj.vertices
                vertices -= np.mean(vertices, axis=0, keepdims=True)
                vertices *= 0.95 / (max(abs(v_min), abs(v_max)))
                obj.vertices = vertices
                return obj

            # total_time = min(nf, total_time)
            vert_datas = []
            v_min, v_max = float("inf"), float("-inf")

            frames = np.linspace(0, nf, total_time, dtype=int, endpoint=False)
            if self.cfg.move_sampling == "first":
                frames = np.linspace(0, min(nf, total_time), total_time, dtype=int, endpoint=False)

            for t in frames:
                vert_data_copy = vert_data
                if t > 0:
                    vert_data_copy = vert_data + offset_data[t - 1]
                vert_datas.append(vert_data_copy)
                vert = vert_data_copy - np.mean(vert_data_copy, axis=0, keepdims=True)
                v_min = min(v_min, np.amin(vert))
                v_max = max(v_max, np.amax(vert))
            grids = []
            for vert_data in vert_datas:
                obj = trimesh.Trimesh(vert_data, face_data)
                obj = normalize(obj, v_min, v_max)
                voxel_grid: trimesh.voxel.VoxelGrid = vox_creation.voxelize(obj, pitch=voxel_size)
                voxel_grid.fill()
                grid = voxel_grid.matrix
                padding_amounts = [(floor((resolution - length) / 2), ceil((resolution - length) / 2)) for length in
                                   grid.shape]
                grid = np.pad(grid, padding_amounts).astype(np.float32)
                grids.append(grid)
            grid = np.stack(grids)
        else:
            mesh: trimesh.Trimesh = trimesh.load(path)
            coords = np.asarray(mesh.vertices)
            coords = coords - np.mean(coords, axis=0, keepdims=True)
            v_max = np.amax(coords)
            v_min = np.amin(coords)
            coords *= 0.95 / (max(abs(v_min), abs(v_max)))
            mesh.vertices = coords
            voxel_grid: trimesh.voxel.VoxelGrid = vox_creation.voxelize(mesh, pitch=voxel_size)
            voxel_grid.fill()
            grid = voxel_grid.matrix
            padding_amounts = [(floor((resolution - length) / 2), ceil((resolution - length) / 2)) for length in
                               grid.shape]
            grid = np.pad(grid, padding_amounts).astype(np.float32)

        # Convert 0 regions to -1, so that the input is -1 or +1.
        grid[grid == 0] = -1

        grid = torch.tensor(grid).float()


        # Doing some sanity checks for 4D and 3D generations
        if self.cfg.mlp_config.params.move:
            assert grid.shape[0] == total_time and grid.shape[1] == resolution and grid.shape[2] == resolution and \
                   grid.shape[3] == resolution
            return grid, 0
        else:
            assert grid.shape[0] == resolution and grid.shape[1] == resolution and grid.shape[2] == resolution

        return grid[None, ...], 0

    def __len__(self):
        return len(self.mesh_files)


class WeightDataset(Dataset):
    def __init__(self, mlps_folder, wandb_logger, model_dims, mlp_kwargs, cfg, object_names=None):
        self.mlps_folder = mlps_folder
        self.condition = cfg.transformer_config.params.condition
        files_list = list(os.listdir(mlps_folder))
        if len(cfg.filter_bad_path) == 0:
            blacklist = {'f3e2df468c15795872517bb0a6b4d3ef', '51ebcde47b4c29d81a62197a72f89474',
                         '46c259e87609c54fafc0cc47720c0ef4', 'a2b758aa5d51642bd32761b337f8b72a',
                         '8ddc3bfd5a86d4be2e7c68eb5d1b9123', 'd6d84d05892a4f492e7c68eb5d1b9123',
                         'dbdca81a0f9079096d511e9563e4bbe7', '3badd7765618bd66a532a2f6f060af39',
                         '3fa5f65a92a5e521d87c63d8b3018b58', '4bfa5d948d9ca7ab7c5f0032facde6fe',
                         '46d2373fa05a4876bc913a3935f0ae10', 'bc16d2cf6420432cb87697d3904b168b',
                         'ce4b8076f8f9a5a05be07e24c1d3227d', 'a61a59a4c48154db37678474be485ca',
                         'daedff5e78136a8b507c9a5cb0f72e1e', 'd2e2e23f5be557e2d1ab3b031c100cb1',
                         '9c7268dd8ec3a3bb590874dcd9dc8481', 'd54a694d514b1911844ac48bcfce34',
                         '29f514bdcf72e779bbf3143b1cb6076a', '5392580fe70da9043554ccf8c30febe7',
                         'c47a2cb459ab849c493ab0b98cb45d3', '67a6b5b12ba64c529a6e43b878d5b335',
                         '947a78a898b6aa81f19a675dcc5ca632', '413a85d9cc7f19a8b6c3e7b944b34fa',
                         '6778c46aff633538c0676369cd1d063d', 'd2bf5f39131584c0a8cba409c4409ba9',
                         '337658edebb67c301ce9f50324082ee4', '5a6eb0f2a316f23666cf1b4a8fc3914e',
                         '2c932237239e4d22181acc12f598af7', '72aedca98ea4429c8ed4db287040dac1',
                         'bc92b144ec7029782e7c68eb5d1b9123', '75705b82eb24f5ae23e79e99b949a341',
                         '2d4a57a467307d675e9e2656aff7dd5b', 'c61f67c3f03b10537f3efc94c2d31dc5',
                         '1c27d282735f81211063b9885ddcbb1', 'a20db17555286e06f5e83e93ffcfd3f0',
                         '33b8b6af08696fdea616caf97d73fa02', 'dba3ab64bab4ed3ed13ef00338ba8c52',
                         '65d7ed8984d78a797c9af13aaa662e8e', '83b55a6e1677563d6ae6891f58c50f',
                         '9f90f3298d7b6c6f938204d34a891739', '71f718d82a865472bfa44fe936def6d4',
                         '65166f18d4ee36a61cac9affde84fd21', '8f4416cb67c3807bcf155ddc002a8f77',
                         'ec879d7360572a5db65bd29086fc2a67', 'c107063f573cbfb3ca8607f540cc62ba',
                         '3c52f7f687ce7efe3df325aac2f73830', 'e06d3e6c1fb4b208cb7c15fd62c3982e',
                         '775120d01da7a7cc666b8bccf7d1f46a', '523f1c2392338cf2b7f9c6f6b7bee458',
                         '6e324581dbbdfb5548e8405d6c51a908', 'fc7387d630c84bb9c863ab010b80d9ed',
                         '3a82056ea319a442f64801ad2940cdd5', '508fa09e7df177e7fee8021c81c84603',
                         '72c28618e3273795f9066cd40fcf015', '5f2aa0bc5a7fc2beacf77e718d93f3e1',
                         '22ce6f6d1adb08d7bbf3143b1cb6076a', '25bd1569261bc545e8323edc0fe816a8',
                         '723c87de224355b69878ac4a791083c5', '4937396b74bc16eaf781741e31f0df4',
                         '65b3e612c00a65a6bc0909d98a1ff2b4', 'b8e4e4994d4674cf2023ec956848b741',
                         '6b84749eaca0e657f37f38dedb2f1219', '220a911e2e303865f64801ad2940cdd5',
                         'cc113b6e9d4fbeb23df325aac2f73830', 'f8ceed6c984895079a6e43b878d5b335',
                         '2ba980d080f89581ab2a0ebad7754fba', 'f9db62e6a88f0d7129343faf3bbffb15',
                         '6d0b546bb6256eb5e66cabd11ba41eae', '794fe891e20800d37bbc9fbc6d1fd31d',
                         '6615bb23e68159c193d4024985440d4c', '28402efe2e3435d124fbfb66ea1f14f7',
                         'a4f5529edc990706e719b5362fe06bbb', '35835c28f33d985fa18e0889038e4fb6',
                         'fbf6917bdd86d5862df404314e891e08', '1d269dbde96f067966cf1b4a8fc3914e',
                         '460cf3a75d8467d1bb579d1d8d989550', '35131f7ea7256373879c08e5cc6e64bc',
                         'd581b2e10f8bc6aecc1d733a19631a1', 'fb62efc64c58d1e5e0d07a8ce78b9182',
                         '580e54df8765aac1c1ca96a73599ca7e', '2aec6e6096e640add00d52e62bf14ee9',
                         '4d50ff789e84e70e54eefcdc602d4520', 'c1b5fd196374203d772a4b403daac29b',
                         '166d333d38897d1513d521050081b441', '3360b510b0408682bbf3143b1cb6076a',
                         'f390b1b28b6dda03dc57b3e43c28d486', '4bd5f77521e76e6a2e690fa6dfd5d610',
                         'f25da5eca572f01bd8d90babf3c5509a', '796bb7d6f4d5ce8471d03b466c72ce41',
                         '460f2b6d8f4dc18d565895440030d853', '7488f87b1e08bbabd00d52e62bf14ee9',
                         'f6f5efa1554038ce2154ead1f7ab03aa', '6dccca8485094190be13ce34aa7c0c1c',
                         'e521828113e1e0c45e28caa3b26a73fd', 'e24be0487f6a3a92d4fd6e0556ecdd3d',
                         '494660cc290492218ac43fbf276bac06', '350d12f5290908c7f446f92b52bbd82a',
                         'ccb9e8dd6453b6c2a2981e27422ad4a7', '7805239ad1e40e07c69d7040c52664c5',
                         'ce2c4502b06c3c356abde8e1529c422f', '53011f6c40df3d7b4f95630cc18536e0',
                         '1b626fd06226b600adcbeb54f3d014e9', '8f33d2c0c08befe48caa71b1fbf7fb98',
                         'dcb5bded8772135b4295343ee7255799', '6a75658fb8242b9c590874dcd9dc8481',
                         '89f21747e6403735d9e1375411b582e', '1df217928b39b927a8cba409c4409ba9',
                         '3390c5050cda83c09a6e43b878d5b335', 'e66692f2ec8ea898874f1daffc45b57c',
                         '934e5dbb193dab924d02f09e7a1a5b28', '3844797c89c5e2e821b85e5214b0d6a7',
                         'd2f8a99bbdc387c8c5552bebbfa48bd7', 'e25e3dc95243a92c59bcb260e54d3785',
                         'd4a8134b2463a8bca8607f540cc62ba', 'c8db76f18f56c1344c2c88971423d0be',
                         '6420a3ff5e526d59e16519c843f95ce0', 'e8bd07a978ac82baef40e4c1c2686cd3',
                         'bf78d2e27a671fce4d4cb1c2a5e48b7a', '7a794db8180858fe90916c8815b5c43',
                         '9e524a14078824b5cfe15db7c5db913', 'e17a696c47d4292393db03f6b4e68f17',
                         'b0b164107a27986961f6e1cef6b8e434', '2f4f38d774e0e941dcc75fd1795fa3a5',
                         '2d255ec7a05b301a203936772104a82d', 'befcb95d80e0e49119ba010ddb4974fe',
                         'be96646f774c7d0e2a23d93958262ccc', '694da337c806f41a7209009cfb89d4bd',
                         '6058d6701a0ca4e748e8405d6c51a908', '6481eb70eb9a58cfb2bb87688c5d2f58',
                         '6fe837570383eb98f72a00ecdc268a5b', '17ad3ab1b1d12a7db26dc8ec64d633df',
                         '95a6c003f5bd8a4acef4e20384a35136', 'aa2af754642256c08699933784576e73',
                         '752a0bb6676c05bbe55e3ad998a1ecb4', '427030abcc0f11a8947bbeb9022263b8',
                         'd56fba80d39bdff738decdbba236bc1d', '3adbafd59a34d393eccd82bb51193a7f',
                         '91bd6e91455f85fddcf9f917545742df', 'bdc5360ff3c62ed69aa9d7f676c1fd7e',
                         '5c72cc06d79edd1cbbf3143b1cb6076a', '94cb05b56b89aa2223895f8c849f85e',
                         '4374a3b4b98e247b398db3ebdf468ed7', 'c58e9f5593a26f75e607cb6cb5eb37ba',
                         'b51032670accbd25d11b9a498c2f9ed5', 'edd9f45c7c927032db5e5b1ef1010d8b',
                         '6ea21a2265075beb9a2f7b9a6f4f875f', '82cd0676627c20b0879eac31511e27a8',
                         '265e6c77443c74bd8043fd2260891a82', 'd583d6f23c590f3ec672ad25c77a396',
                         'd63bd140efa537dcf73e5bc170b7e2be', 'f9e80ce23d9536623fddedb0bf24c68a',
                         '2c5bdd9a08122f9e2023ec956848b741', '4b20c1736440ff9d90dd6eb126f6cbbb',
                         '194098fb0aefc4a0666bd6da67d3abc2', 'be11ce096084bd784f95630cc18536e0',
                         'a61d4ad108af6eecca8607f540cc62ba', '2cc1ff07bcb27de4f64801ad2940cdd5',
                         'f96426f2392abb1d8d58389384d9812e', '395afa94dd4d549670e6bd9d4e2b211f',
                         'ffdaed95c3063cdf3ce4891c7dcdfb1c', '8bfcf5618f2d169c9a6e43b878d5b335',
                         '94d3a666d2dbc4385ff3feb917a6004b', 'c5e04f903e3472c31883411175051361',
                         'a9cdbca070047fe61e9dc95dd6a9f6', '9dbc0aba1311e7b8663e90eaf6b4ca52',
                         'a762fe69269fc34b2625832ae05a7344', '157a81baeb10914566cf1b4a8fc3914e',
                         'f7110ecac70994a83820d8f180caa23a', 'cbb5ed444249f0a9e90916c8815b5c43',
                         'dc7c5d12854b9467b96212c8f6cd06e', 'ba37c8ef604b675be1873a3963e0d14',
                         'a591b49406ab0d2abbf3143b1cb6076a', 'ebe0d0bfa6ec36edd88eab18f1be033b',
                         '9617ea33f17993e76f0c490da56d34b', '9baf5b31d70e0d05e98d814cc4d9c5e3',
                         '7e75688f4b185d4193a78ffd70737098', 'bee504b4909890df1dfabee9ba27dc70',
                         'c6bcec892954942a83855ba2afe73b0b', '2c9797204c91e3a440975e4feec771f6',
                         'a60b2775f1955153ca8607f540cc62ba', '552871abb864d06d35fc197bbabcd5bd',
                         'ce337df2f75801eeb07412c80bd835', '5aeb583ee6e0e4ea42d0e83abdfab1fd',
                         '880715a3ef69f47e62b600da24e0965', '873f4d2e92681d12709eb7790ef48e0c',
                         '8ec085a86e6d9425f4fb6842b3610149', '2a06adfb446c85c9f9d3f977c7090b2a',
                         'afd43430bd7c689f251fe573763aebef', '1eb3af39d93c8bf2ddea2f4a0e0f0d2e',
                         'eae54f8d47cf07c0aeec93871310e650', 'd75a4d4d25863f5062747c704d78d4f8',
                         '4e4128a2d12c818e5f38952c9fdf4604', 'af6b292c7857c78abb0e5c1799dab683',
                         'b87185699318f4b635fc197bbabcd5bd', 'bd22bcf18a9c8163adecb6fc00604132',
                         '70ed0fe305145f60e53236e6f2fbeb94', '4f1fb7c062c50fb15a2c5766752aea65',
                         '4f0bf26c62bb7c8b7e1c97634acf0214', '17e66cd463ff0de126360e1e29a956c7',
                         'c31f5303997022c842c20743f866e1a6', '3265b621ca222d29d00d52e62bf14ee9',
                         '41aafedd84a6fa7490baeef8ba5b93e5', '77ab8bb69221b13bbc0909d98a1ff2b4',
                         '802cbaaf1a51cf38c863ab010b80d9ed', 'e25794343ee37d6fa8eeb11153b68d81',
                         '62ebe2dd7bceccd097f28f82dd9c77a2', 'ade3c4987f49895ff960bc420d751255',
                         '755b0ee19aa7037453e01eb381ca65', '4e2322d4f1c0d29df96e777471c18dbe',
                         '1560968d05cd8887cc14f1e6f4f4f49b', 'c9584d90a1da19f723a665a253ac8cae',
                         'c85e3f6c572581de7d3b11085e75c7ad', '10aa040f470500c6a66ef8df4909ded9',
                         'd532a5abb756ebebcc14f1e6f4f4f49b', 'cf4c2d7d836c781e5a59e179e114b9a7',
                         'a1b95d2e1264f6bd66ccbd11a6ef0f68', 'e161df613fc808b0d7ec54df5db5828c',
                         '530d0edd6e2881653023dc1d1218bb2d', 'b3fbc7a0b0e3a821fd279055f27928f7',
                         'cc9b7118034278fcb4cdad9a5bf52dd5', 'cb8fb401a278fc36bbf3143b1cb6076a',
                         'c2d5bd1215248f9c8b6c29bda2bc905a', '9436273fc1a5e3ca7af159eaf7625abf',
                         'b5cdecafe353e18ac1006ed55bc1a3fc', '121b5c1c81aa77906b153e6e0582b3ac',
                         'f50eba69a3be1a1e536cfc00d8c31ac5', '7e8be9c88e596b84198609c994ea801',
                         'a051219f3f444fadc5e2bf7b5a5f1c56', '69d7ae2143c2a2709c6df67fce5fa25a',
                         '74ebf601d9c872a7828204947d78b9af', '92a83ecaa10e8d3f78e919a72d9a39e7',
                         'f5a8cae96024e709c1ccec171a275967', '8baeb2c664b0bdf4ca8607f540cc62ba',
                         'd3dcf83f03c7ad2bbc0909d98a1ff2b4', '1280f994ba1f92d28699933784576e73',
                         'a849d4325008fbea85dfb1711fe4ff6d', 'fc0dbd045c9391ce4a29fef5d2b2f3d7',
                         'b943b632fd36f75ac1ccec171a275967', 'e9df546c1e54af0733d72ae4e2c33411',
                         '57f1dfb095bbe82cafc7bdb2f8d1ea84', '1d0c128592385580e2129f6359ec27e3',
                         '844d36a369cdeed3ac4f72bf08dc79a6', 'dd48aa92170bdde04c3a35cee92bb95b',
                         'fb402a36e91ab1b44e7761521d3c6953', '1890f6391df25286394b1e418d5c594',
                         'fbb2e9c15888afcaca504cdc40c452de', '22944fabf8a9763f28132d87f74ffe13',
                         '71bb720c33da689090b1d6deb98feec6', 'efc7d4fb87937413dc13452e3008005b',
                         '389dd6f9d6730b0e29143caa6b05e24f', '13370c42b8ea45cf5e8172e6d9ae84ff',
                         '812111e3a4a545cbc863ab010b80d9ed', '1dbcb49dfbfd0844a480511cbe2c4655',
                         '150cdc45dabde04f7f29c61065b4dc5a', 'dfa36bffe436a98ee0534173b9189765',
                         '1d4ff34cdf90d6f9aa2d78d1b8d0b45c', 'e2275ee8d6b175f2f446f92b52bbd82a',
                         '1f672d2fd5e3f4e78026abe712c1ab05', 'fb0f2907b977e7cb67c5e3041553656b',
                         '6da4668de7ccdd0d4d10a13d437fced6', '66cd9502b63e8a97bbf3143b1cb6076a',
                         '7d180493022c01daace5721ccacba16', 'b2bb5a56b3d805b298b8c800ae001b66',
                         '828176e6eaee542ceb532c8487004b3c', '83dd9dd45724d5fbbeb310a83b693887',
                         'ed4aaf81dc577bedac4f72bf08dc79a6', 'eb7bf553e500b9b544bf3710e93f8cf7',
                         '3f69370401d4dc9a275386e1d3ac388e', '81596cc38eef8260ce9e5ac45c67ec22',
                         'ab35aa631852d30685dfb1711fe4ff6d', '9300dc1ca5f16b074f95630cc18536e0',
                         '6ba7cad8fa7301f9c1ca96a73599ca7e', 'fbd800d43c5f0d74250cb4f7fcd9ec03',
                         '7e1d4583f70c8583431589819e5ca60c', '8448475504ba9fdeca8607f540cc62ba',
                         'bfd606459cace196e7ee2e25a3cfaa4d', 'faa361f642620bb72def37e9c0b35d0e',
                         'e4ac77fdf28116232fa725d7a62a02a', '19604020a86ab1790b1d6deb98feec6',
                         '40192d0e50b4d2c1f27a705edb2f9ba6', '5f46f3c62e353c7bb4f5fdc59ce06e88',
                         '35e2eceef33804d8196c5820729d438f', 'e4e1b542f8732ae1c6768d9a3f27965',
                         'a2c5e769f19c7e97b7d7aa9b9ebcccb0', '5b048655453b37467584cbfee85fb982',
                         '95e589163afd0a7a609e2d916fa0da27', 'ec8ba88cdead53f336dafa9b6763ef3f',
                         'cf71f5442c4120db37678474be485ca', '3cd1b98f12d2a22bf3ad4b0977199f23',
                         '6dedeb5b87ee318b2154ead1f7ab03aa', '7442ad61d59fac8c7372dc0a2f1067b1',
                         '85396c57cdbf919f83467b3671ddaea2', 'c1260a89eee28413f2acf00738ce9d0d',
                         '189f045faacc1b5f9a8993cdad554625', 'eaa0d465e9d0c16acfbf0f1430c86945',
                         '6ce399fe42d54815e4406b3bf37e6ffe', 'e841e17e3256acf38699933784576e73',
                         'f6373cc88634e8ddaf781741e31f0df4', '36dd57178402cdf2afd477f714c68df9',
                         '8a674703723db7a390baeef8ba5b93e5', '6b6cb0c71731aacc277d303e3a640f98',
                         'e523ba4e79a48d31bd46d022fd7d80aa', 'de1a7b4e9911e48b48e8405d6c51a908',
                         '2cc44e0f9fb1efdb85e0a2007a11e92f', '6509073d1ff37d683d41f76be7f2e91f',
                         '9c7395d87c59aa54a79f2ed56427c6e6', 'a3e15e215144dda0a03ebab0e8b8f7a0',
                         '8f40518bd30467151e5ae32cb9e3711f', '70e4200e848e653072ec6e905035e5d7',
                         'f89b085c37779a5997517c313146c4ab', '92b7d0035cefb816d13ef00338ba8c52',
                         '795e0051d9ce7dfe384d4ad42dbd0045', '87fb26b8e56d25f2b87697d3904b168b',
                         '17c86b46990b54b65578b8865797aa0', '84b396dde837c81994445a3e8b9de59d',
                         'd16405b7a4300014ef5bed0139d3780c', 'ddfbeb997ef83cab884a857d19f7439f',
                         '4fccf49d6356c756b833a96759a020e2', 'b31bbc50a0d3a4366cf1b4a8fc3914e',
                         'fc5dade8622f686b4aba1f0cb15b1439', '30b514b24624da4fc1ccec171a275967',
                         '25a057e935aeb4b6842007370970c479', '519f1ddcbf942a76a71b0c9b506dc672',
                         'f59a2be8fd084418bbf3143b1cb6076a', '5cbe5be753b5c7faf389d19fad321c37',
                         '5ae05c956af61890b58b3ab5dbaea0f7', '4eced94670d10b35e856faf938562bd0',
                         '396312e9bec88c2590b1d6deb98feec6', 'd3f93b6da62fae46a98ae8c73b190485',
                         '8eeb9f69fc9ef1b0b45fd154bd3b6957', 'af69c8020fa9b68366cf1b4a8fc3914e',
                         '4c880eae29fd97c1f9575f483c69ee5', '909f59399d056983a0a3307f8f7721fc',
                         'd45772f938d14d52736e55e9ba083986', 'f7c11b5e07e9ccab3a116bc3efac4e3b',
                         '57a57f639a3e636d914c075742032f6', '96600318559071d48caa71b1fbf7fb98',
                         'fcd7a8834a7f26f15069db69b8c1c70', 'eed299b690be51ffbd931fcaa69140',
                         '6cf339eb8c950ac5d556fc099c90ab45', 'aeb538b2f1c36a8d9e811b082458229e',
                         '29120728132e5dce42a2048a31b7df8c', '4e4ae13865bf47f41adbb4c4e06ad649',
                         'ecbb6df185a7b260760d31bf9510e4b7', 'afd02e6d4cf0a342c1ccec171a275967',
                         '751b1e75fcd7f1deffb814dfce3ab22e', 'fbebcde2d8fbf81ee7cf320ab5761d45',
                         '757c47e20a37647431e38f024b7ad042', 'a87adc4fb1229b7f6d0f2528dd6f85b9',
                         'ad6e93a1db3e1da5977e4bb19a62128e', '393cfa7e090b972dce2cec85ea6ae00d',
                         'c950fc7d559f30016e86a8ae6e1f4d7e', 'd390f0246fd43cc8bd46d022fd7d80aa',
                         'cb1aff81a3c6ef71d25fd5e3115979a5', '381111f176565d48fe4c91be246ef13b',
                         'c3733e626c07b9ff26360e1e29a956c7', '800334df5da57266a4642ec4b6f68a',
                         'b9e6298004d7d422bd46d022fd7d80aa', 'f8647af0b1ad01445de04aad18bd94c3',
                         'd0001af4b3c1fe3d6f675e9f2e677792', 'db73a3c857949f469a6e43b878d5b335',
                         '62fe06fd4f1b390fa9bcc7eaa4032fa4', '959044f10e27b89ee664ce1de3ddc8b4',
                         'e58010dd5766e0ce78f081615c34707c', '299ec43108d7109113ae47e860a2333a',
                         'f562ff06e51e573e42979ff355194f16', 'abbe69a6f94918c79eb9aa3111a82815',
                         '37f2f187a1582704a29fef5d2b2f3d7', 'a3c928995562fca8ca8607f540cc62ba',
                         '7f895411162624e92023ec956848b741', 'c0c32558decf271df3ad4b0977199f23',
                         'ebedcd06f1770cd4bbf3143b1cb6076a', 'b702e35f4a59e81f64801ad2940cdd5',
                         'ee92bc35ee989f59271b3fb2659dec56', '8b72934186e1d8b0f510cd52a5f27547',
                         'ed0a9a32a8e35f21ca8607f540cc62ba', '48b99ae8fbe0762a8ed04761eced33c6',
                         '697b269a890237fe15796a932d10290d', 'f80343ac3064e74862347b4731688b0f',
                         '2893dc61be63a8a16d0ff49003c479bc', 'db4079b8c7d3d674ca8607f540cc62ba',
                         'd30689ca6cdf2601f551b6c3f174499e', 'd940f33afc01ff036da97d9f744f7e97',
                         '157bb84c08754307dff9b4d1071b12d7', 'e5a7a353d5fa8df844b2fa2cac0778f5',
                         '9483e1b0e4222cb4f2b0736dd4d8afe0', '965d457720def9a490b1d6deb98feec6',
                         'dd9a7dd5e2ea3389938204d34a891739', 'a2041f74f316e7b9585e3fa372e910b7',
                         'a361d82b2c510ca5208842e3d616cb23', '68c61d42222863682296d88107d065f6',
                         '3ac64a2c67cb44f19777d69c8d47140', 'e88e090caa1ccc5d187bd96066d7269e',
                         '4bb41171f7e6505bc32f927674bfca67', '12a1ac26d29ed1083554ccf8c30febe7',
                         '21827b0be78dd3e17dd9ca9e282b9209', '7175100f99a61c9646322bce65ca3756',
                         'b7b743834a6d78c2225a23c790f08fdd', '6ba642ca477a73db4c3a35cee92bb95b',
                         '2026699e25ba56c5fd6b49391fda17', '3f9cab3630160be9f19e1980c1653b79',
                         '4982bea0a007c19593b2f224b3acb952', '1ce5b9a79420e946bff7790df3158906',
                         'd172705764e25e20884a857d19f7439f', '52712e1c07ea494419ba010ddb4974fe',
                         'bc58ff3369054fa68f52dc705c3109b9', 'ed57671fc0252e15b95e9a91dc6bad16',
                         '7f837b389e885af471b4c018296f73c7', '1286826ff37699a1a0d713eb26379316',
                         'c0f9c28c45e7c8354f95630cc18536e0', 'd43b80dd95a2233a5ae839ffe09b9d31',
                         '22c11b2bab2cf93fc1ccec171a275967', '53edcc6832e776dcca8607f540cc62ba',
                         'ce682d7a2bbf77b6fc4b92d3d335214a', '839a950d0264cbb89a162c818d22a620',
                         '19a624cf1037fc75cda1835f53ae7d53', 'e9bae38bc2083d0bb4d73e4449062b04',
                         'f144e93fe2a11c1f4c3a35cee92bb95b', 'd109c08886c2a3dabdf566b587d6b21',
                         'e9f39176973edd33a8cba409c4409ba9', 'fbe788465e564e46bc0909d98a1ff2b4',
                         '85da8ecc055fc6cb58328b65a4733701', 'cbf4dc1c144ce656ffa79951b9f955a3',
                         '72a74e13c2424c19f2b0736dd4d8afe0', 'a287dc5d0e28d3d3325212819caa597d',
                         'b04ec55f4960b3b984b7ea000aa0a2b', '1a963a929d9b1332290d63dca780cfb6',
                         'b82731071bd39b66e4c15ad8a2edd2e', 'f13827d156628467b4cdad9a5bf52dd5',
                         '369244d49f8f1308b858e64ff0fa8db3', '14cd2f1de7f68bf3ab550998f901c8e1',
                         '7ee59463dc17ac6e3e3f3c9608255377', 'd80afa36aeb72c552b5147716975ed8a',
                         '98168c1772b769c0ea1bd6f2443b43e7', 'e452189bb7bd6617ef7cbef6334698fc',
                         '2b96f4b4896962473eb731541f9f8d', 'cb7c32bd7266daef37f38dedb2f1219',
                         '253a1aead30731904c3a35cee92bb95b', '8eda6d2dcf9302d2d041917556492646',
                         '6ad89740605331aef5f09964a6a1f97', 'e559a626d0ef8b4f982014dd9aabdeeb',
                         'da67955425ffe66071d03b466c72ce41', 'b80bd34ab330babbc8727b27ee96a4b7',
                         '95cfdf728da16975c5f6fdebb053ab2f', '78bd38a7282a73f8b184ba15dd506a2d',
                         'd78a16856adad344670aaa01f77ae41a', '947d6b9cd1966e2e719b5362fe06bbb',
                         '4c9214d70e0a00c6c1ccec171a275967', '5fed73635306ad9f14ac58bc87dcf2c2',
                         'e8c1e738997275799de8e648621673e1', '5f9b4ffc555c9915a3451bc89763f63c',
                         '5a37bc42a52130a18f52dc705c3109b9', 'c3408a7be501f09070d98a97e17b4da3',
                         '1f08b579e153b2de313f9af5275b7c70', 'ea58a51483e3604897dec65c2238cb8a',
                         '446f9144536c0e47f0c7ca80512a3ddf', '464a8718f0e81ffd9a6e43b878d5b335',
                         '2e3c317357ecb038543941eaaf04581f', 'd38922599bc74f6da30fd8ce49679098',
                         'e0a8ae255ed47518a847e990b54bf80c', '7addd02b1c255edcc863ab010b80d9ed',
                         '2628b6cfcf1a53465569af4484881d20', '85d3691b7bde76548b96ae1a0a8b84ec',
                         'a4ea22087dec2f32c7575c9089791ff', 'e954dc13308e6756308fc4195afc19d3',
                         'cc40acee83422fe892b90699bc4724f9', '4e67529b0ca7bd4fb3f2b01de37c0b29',
                         '34e87dd1c4922f7d48a263e43962eb7', '92e445da194d65873dc5bf61ec5f5588',
                         '64211a5d22e8ffad7209009cfb89d4bd', 'aaefbfb4765df684cf9f662004cc77d8',
                         '414f3305033ad38934f276985b6d695', 'c854bf983f2404bc15d4d2fdca08573e',
                         '2a3d485b0214d6a182389daa2190d234', '117830993cc5887726587cb13c78fb9b',
                         'a2491ac51414429e422ceeb181af6a7f', 'bcaf04bfae3afc1f4d48ad32fb72c8ce',
                         '8a84a26158da1db7668586dcfb752ad', '556363167281c6e486ecff2582325794',
                         '6d752b942618d6e38b424343280aeccb', '8b59ed9a391c86cdb4910ab5b756f3ae',
                         'd24f2a1da14a00ce16b34c3751bc447d', 'b5d0ae4f723bce81f119374ee5d5f944',
                         '542a1e7f0009339aa813ec663952445c', 'ae8a5344a37b2649eda3a29d4e1368cb',
                         'e2a6bed8b8920586c7a2c209f9742f15', '237b5aa80b3d3461d1d47c38683a697d'}
        else:
            blacklist = set(np.genfromtxt(cfg.filter_bad_path, dtype=str))
        if object_names is None:
            self.mlp_files = [file for file in list(os.listdir(mlps_folder))]
        else:
            self.mlp_files = []
            for file in list(os.listdir(mlps_folder)):
                # Excluding black listed shapes
                if file.split("_")[1] in blacklist and cfg.filter_bad:
                    continue
                # Check if file is in corresponding split (train, test, val)
                # In fact, only train split is important here because we don't use test or val MLP weights
                if ("_" in file and file.split("_")[1] in object_names) or (file in object_names):
                    self.mlp_files.append(file)
        self.transform = None
        self.logger = wandb_logger
        self.model_dims = model_dims
        self.mlp_kwargs = mlp_kwargs
        if cfg.augment in ["permute", "permute_same", "sort_permute"]:
            self.example_mlp = get_mlp(mlp_kwargs)
        self.cfg = cfg
        if "first_weight_name" in cfg and cfg.first_weight_name is not None:
            self.first_weights = self.get_weights(
                torch.load(os.path.join(self.mlps_folder, cfg.first_weight_name))).float()
        else:
            self.first_weights = torch.tensor([0])

    def get_weights(self, state_dict):
        weights = []
        shapes = []
        for weight in state_dict:
            shapes.append(np.prod(state_dict[weight].shape))
            weights.append(state_dict[weight].flatten().cpu())
        weights = torch.hstack(weights)
        prev_weights = weights.clone()

        # Some augmentation methods are available althougwe don't use them in the main paper
        if self.cfg.augment == "permute":
            weights = random_permute_flat([weights], self.example_mlp, None, random_permute_mlp)[0]
        if self.cfg.augment == "sort_permute":
            example_mlp = generate_mlp_from_weights(weights, self.mlp_kwargs)
            weights = random_permute_flat([weights], example_mlp, None, sorted_permute_mlp)[0]
        if self.cfg.augment == "permute_same":
            weights = \
                random_permute_flat([weights], self.example_mlp, int(np.random.random() * self.cfg.augment_amount),
                                    random_permute_mlp)[0]
        if self.cfg.jitter_augment:
            weights += np.random.uniform(0, 1e-3, size=weights.shape)

        if self.transform:
            weights = self.transform(weights)
        # We also return prev_weights, in case you want to do permutation, we store prev_weights to sanity check later
        return weights, prev_weights

    def __getitem__(self, index):
        file = self.mlp_files[index]
        dir = join(self.mlps_folder, file)
        if os.path.isdir(dir):
            path1 = join(dir, "checkpoints", "model_final.pth")
            path2 = join(dir, "checkpoints", "model_current.pth")
            state_dict = torch.load(path1 if os.path.exists(path1) else path2)
        else:
            state_dict = torch.load(dir, map_location=torch.device('cpu'))

        weights, weights_prev = self.get_weights(state_dict)

        if self.cfg.augment == "inter":
            other_index = np.random.choice(len(self.mlp_files))
            other_dir = join(self.mlps_folder, self.mlp_files[other_index])
            other_state_dict = torch.load(other_dir)
            other_weights, _ = self.get_weights(other_state_dict)
            lerp_alpha = np.random.uniform(low=0, high=self.cfg.augment_amount)  # Prev: 0.3
            weights = torch.lerp(weights, other_weights, lerp_alpha)

        return weights.float(), weights_prev.float(), weights_prev.float()

    def __len__(self):
        return len(self.mlp_files)