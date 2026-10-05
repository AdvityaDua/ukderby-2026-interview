import asyncio
import websockets
import json
import base64
import os
import subprocess

async def test_interview():
    uri = "ws://127.0.0.1:8000/api/v1/interview/chat/test-client-123?token=mock_token_not_checked_anyway"
    
    print(f"Connecting to {uri}...")
    try:
        async with websockets.connect(uri) as websocket:
            print("Connected! Sending initialization payload...")
            
            init_payload = {
                "type": "init",
                "company": "Google",
                "role": "Backend Engineer",
                "interview_type": "technical",
                "jd_text": "We need a strong backend engineer who knows Python and system design."
            }
            await websocket.send(json.dumps(init_payload))
            
            # Wait for init response
            while True:
                msg = await websocket.recv()
                data = json.loads(msg)
                if data.get("type") == "info" and data.get("content") == "Context initialized.":
                    print("Context initialized successfully!")
                    break
                print(f"Server: {data}")
            
            # Now simulate the user saying something
            print("\n--- Sending User Message ---")
            await websocket.send(json.dumps({
                "type": "message",
                "content": "Hi, I am ready for the interview!"
            }))
            
            print("\n--- Receiving Streamed Response ---")
            audio_idx = 0
            
            while True:
                msg = await websocket.recv()
                data = json.loads(msg)
                
                if data.get("type") == "text":
                    print(data["content"], end="", flush=True)
                
                elif data.get("type") == "audio":
                    print(f"\n\n[Received Audio Chunk {audio_idx}! Playing automatically...]\n")
                    # Save base64 to MP3 file
                    b64_data = data["base64"]
                    audio_bytes = base64.b64decode(b64_data)
                    filename = f"chunk_{audio_idx}.mp3"
                    with open(filename, "wb") as f:
                        f.write(audio_bytes)
                    
                    # Play the audio asynchronously using afplay (built-in macOS command)
                    subprocess.Popen(["afplay", filename])
                    
                    audio_idx += 1
                
                elif data.get("type") == "stream_end":
                    print("\n\n--- End of Stream ---")
                    break
                
                elif data.get("type") == "error":
                    print(f"\nError: {data}")
                    break
                    
    except Exception as e:
        print(f"WebSocket Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_interview())