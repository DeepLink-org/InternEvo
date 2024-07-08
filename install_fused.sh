if [ "$1" = "uninstall" ]; then
  pip uninstall -y flash_attn rotary-emb xentropy-cuda-lib apex
  exit
fi

cd ./third_party/flash-attention
python setup.py install
cd ./csrc
cd xentropy && pip install -v .
cd ../rotary && pip install -v .
cd ../../../../
cd ./third_party/apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
cd ../../
