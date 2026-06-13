FROM lmsysorg/sglang@sha256:a081e28f8d6368de246ae66459efbc6f94ea20ffa15fe3f3ce6b0820e9b58eea

COPY scripts/patch_loader.py /opt/minimax-m3/patch_loader.py
RUN python3 /opt/minimax-m3/patch_loader.py \
  /sgl-workspace/sglang/python/sglang/srt/models/minimax_m3_vl.py

ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

ENTRYPOINT ["python", "-m", "sglang.launch_server"]
