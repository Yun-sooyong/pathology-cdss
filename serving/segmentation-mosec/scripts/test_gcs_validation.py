import msgpack, requests

url = "http://127.0.0.1:8000/inference"
headers = {"Content-Type": "application/msgpack"}

# case 1: gcs_image_path 필드 자체가 없는 경우
resp = requests.post(url, data=msgpack.packb({"job_id": "test-1"}), headers=headers)
print("case1:", resp.status_code, resp.content)

# case 2: gcs_image_path는 있는데 형식이 틀린 경우
resp = requests.post(
    url,
    data=msgpack.packb({"job_id": "test-2", "gcs_image_path": "not-a-gcs-path"}),
    headers=headers,
)
print("case2:", resp.status_code, resp.content)