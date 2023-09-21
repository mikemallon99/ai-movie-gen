git clone https://github.com/justinjohn0306/Wav2Lip wav2lip_model

#download the pretrained model
wget 'https://github.com/justinjohn0306/Wav2Lip/releases/download/models/wav2lip.pth' -O 'wav2lip_model/checkpoints/wav2lip.pth'
wget 'https://github.com/justinjohn0306/Wav2Lip/releases/download/models/wav2lip_gan.pth' -O 'wav2lip_model/checkpoints/wav2lip_gan.pth'
wget 'https://github.com/justinjohn0306/Wav2Lip/releases/download/models/resnet50.pth' -O 'wav2lip_model/checkpoints/resnet50.pth'
wget 'https://github.com/justinjohn0306/Wav2Lip/releases/download/models/mobilenet.pth' -O 'wav2lip_model/checkpoints/mobilenet.pth'
pip install https://raw.githubusercontent.com/AwaleSajil/ghc/master/ghc-1.0-py3-none-any.whl
pip install git+https://github.com/elliottzheng/batch-face.git@master
