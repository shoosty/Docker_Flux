## Volume Mount Points
- Persistent Pod: /workspace
- Serverless: /runpod-volume
- Same volume, same data, different mount path depending on pod type
- Handler.py uses /runpod-volume (serverless only)
- To browse/download models: use a persistent pod, look in /workspace
