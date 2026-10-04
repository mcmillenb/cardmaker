import base64
import os

import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
from insightface.app.common import Face
from flask import Flask, request, jsonify

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'faceswap-models', 'inswapper_128_fp16.onnx')

# A photo whose face is less similar than this (cosine, ArcFace embeddings) to the
# average of the other photos is treated as a different person and left out of the identity.
# Same-person photos typically score well above 0.4; strangers sit near 0.
OUTLIER_SIMILARITY = 0.3

# The detector runs at a permissive threshold and callers filter by det_score:
# photos and automatic swaps want confident faces, but when the user has marked a
# face by hand, a low-confidence hit inside the mark (painted faces often score
# 0.2-0.5) is far better than nothing.
DET_THRESH = 0.2
CONFIDENT_SCORE = 0.5
MARKED_SCORE = 0.3
SWAP_PASSES = 2

# ArcFace's canonical 5-point layout (eyes, nose, mouth corners) in a 112x112 crop.
# Used to place a face into a marked region the detector can't see at all.
ARCFACE_KPS = np.array([
    [38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
    [41.5493, 92.3655], [70.7299, 92.2041],
], dtype=np.float32)

# Loaded once at process startup so requests are fast (~1s) instead of paying
# model-load cost (~10s) on every call.
analyzer = FaceAnalysis(name='buffalo_l')
analyzer.prepare(ctx_id=0, det_size=(640, 640), det_thresh=DET_THRESH)
swapper = insightface.model_zoo.get_model(MODEL_PATH, download=False, providers=['CPUExecutionProvider'])

app = Flask(__name__)


def decode_image(b64):
    raw = base64.b64decode(b64.split(',', 1)[-1] if ',' in b64 else b64)
    arr = np.frombuffer(raw, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def encode_image(img):
    ok, buf = cv2.imencode('.png', img)
    return base64.b64encode(buf.tobytes()).decode('ascii')


def face_area(face):
    x0, y0, x1, y1 = face.bbox
    return (x1 - x0) * (y1 - y0)


def detect(img, min_score=CONFIDENT_SCORE):
    return [f for f in analyzer.get(img) if f.det_score >= min_score]


def center_dist2(face, cx, cy):
    fx = (face.bbox[0] + face.bbox[2]) / 2
    fy = (face.bbox[1] + face.bbox[3]) / 2
    return (fx - cx) ** 2 + (fy - cy) ** 2


def find_marked_face(img, cx, cy, rx, ry):
    """Finds the face the user marked, trying progressively harder.

    Returns (face, approximate). Faces in generated art are often only ~50px tall,
    which is near the detector's floor, so after a full-image pass this zooms into
    the marked area (upscaled to the detector's 640px input) and tries again. If even
    that finds nothing, it fits a canonical face layout to the marked oval, which
    swaps well as long as the face is roughly upright and facing the viewer.
    """
    reach = max(rx, ry, 24)

    def inside_mark(f):
        return center_dist2(f, cx, cy) <= (reach * 1.5) ** 2

    near = [f for f in detect(img, MARKED_SCORE) if inside_mark(f)]
    if near:
        return min(near, key=lambda f: center_dist2(f, cx, cy)), False

    h, w = img.shape[:2]
    half = int(min(max(reach * 2.5, 64), max(w, h)))
    x0, y0 = max(0, int(cx - half)), max(0, int(cy - half))
    x1, y1 = min(w, int(cx + half)), min(h, int(cy + half))
    crop = img[y0:y1, x0:x1]
    if crop.size:
        scale = 640 / max(crop.shape[:2])
        zoomed = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        faces = analyzer.get(zoomed)
        for f in faces:
            f.bbox = f.bbox / scale + np.array([x0, y0, x0, y0], dtype=np.float32)
            f.kps = f.kps / scale + np.array([x0, y0], dtype=np.float32)
        near = [f for f in faces if inside_mark(f)]
        if near:
            return min(near, key=lambda f: center_dist2(f, cx, cy)), False

    # The ArcFace layout's face spans roughly 72px wide x 90px tall (brow to chin)
    # of its 112px crop, centered near (56, 66).
    s = (2 * max(rx, 12) / 72 + 2 * max(ry, 15) / 90) / 2
    kps = (ARCFACE_KPS - np.array([56, 66], dtype=np.float32)) * s + np.array([cx, cy], dtype=np.float32)
    face = Face(bbox=np.array([cx - rx, cy - ry, cx + rx, cy + ry], dtype=np.float32), kps=kps, det_score=0.0)
    return face, True


def normalize(v):
    return v / np.linalg.norm(v)


@app.get('/health')
def health():
    return {'ok': True}


@app.post('/identity')
def identity():
    """Builds one identity from several photos of the same person.

    Each photo contributes the normed embedding of its largest face; the identity is
    their (re-normalized) mean, which is steadier than any single photo's lighting,
    angle or expression. inswapper only consumes the embedding, so this is all the
    swap needs — the frontend keeps it and sends it back with every /swap.
    """
    data = request.get_json(force=True)
    photos = data.get('photos') or []
    results = []
    embeddings = []
    traits = []
    for i, b64 in enumerate(photos):
        img = decode_image(b64)
        if img is None:
            results.append({'index': i, 'status': 'unreadable'})
            continue
        faces = detect(img)
        if not faces:
            results.append({'index': i, 'status': 'no-face'})
            continue
        results.append({'index': i, 'status': 'ok'})
        face = max(faces, key=face_area)
        embeddings.append((i, face.normed_embedding))
        traits.append((i, face.age, face.sex))

    if not embeddings:
        return jsonify({'error': 'No face was detected in any of the photos. Try clearer, front-facing photos.', 'photos': results}), 400

    vectors = [e for _, e in embeddings]
    mean = normalize(np.mean(vectors, axis=0))
    if len(embeddings) >= 3:
        # Compare each photo against the average of the *other* photos, so a stray
        # photo of someone else can't pull the reference toward itself.
        kept = []
        total = np.sum(vectors, axis=0)
        for i, e in embeddings:
            similarity = float(np.dot(e, normalize(total - e)))
            if similarity < OUTLIER_SIMILARITY:
                results[i] = {'index': i, 'status': 'different-person', 'similarity': similarity}
            else:
                kept.append(e)
        if kept:
            mean = normalize(np.mean(kept, axis=0))

    used = sum(1 for r in results if r['status'] == 'ok')
    # Rough age/sex from the photos that were kept, used only to pre-fill the
    # editable description the art prompt is built around.
    kept_traits = [(age, sex) for i, age, sex in traits if results[i]['status'] == 'ok']
    ages = [age for age, _ in kept_traits if age is not None]
    sexes = [sex for _, sex in kept_traits if sex]
    return jsonify({
        'embedding': mean.tolist(),
        'used': used,
        'photos': results,
        'age': round(float(np.median(ages))) if ages else None,
        'sex': max(set(sexes), key=sexes.count) if sexes else None,
    })


def frontalness(face):
    """How far the nose sits from the midpoint between the eyes, as a fraction of
    the eye distance: ~0 facing the viewer, ~0.5+ in three-quarter or profile view."""
    left_eye, right_eye, nose = face.kps[0], face.kps[1], face.kps[2]
    eye_dist = np.linalg.norm(right_eye - left_eye)
    if eye_dist < 1:
        return 1.0
    return float(abs(nose[0] - (left_eye[0] + right_eye[0]) / 2) / eye_dist)


@app.post('/inspect')
def inspect():
    """Reports the largest confident face in an image, so the art generator can
    decide whether a render has a face worth swapping onto or should be retried."""
    data = request.get_json(force=True)
    faces = detect(decode_image(data['image']))
    if not faces:
        return jsonify({'face': None})
    face = max(faces, key=face_area)
    x0, y0, x1, y1 = (float(v) for v in face.bbox)
    return jsonify({'face': {
        'bbox': [x0, y0, x1, y1],
        'height': y1 - y0,
        'score': float(face.det_score),
        'yaw': frontalness(face),
    }})


@app.post('/swap')
def swap():
    data = request.get_json(force=True)
    target_img = decode_image(data['target'])
    region = data.get('region')

    if data.get('embedding'):
        embedding = np.asarray(data['embedding'], dtype=np.float32)
        if embedding.shape != (512,):
            return jsonify({'error': 'Face identity has the wrong shape; rebuild it from the photos.'}), 400
        source_face = Face(embedding=embedding)
    else:
        source_faces = detect(decode_image(data['source']))
        if not source_faces:
            return jsonify({'error': 'No face detected in the uploaded photo. Try a clearer, front-facing photo.'}), 400
        source_face = max(source_faces, key=face_area)

    approximate = False
    if region:
        target_face, approximate = find_marked_face(
            target_img, region['cx'], region['cy'], region.get('rx') or 0, region.get('ry') or 0,
        )
    else:
        target_faces = detect(target_img)
        if not target_faces:
            return jsonify({'error': 'No face detected in the generated art, so there is nothing to swap onto. Drag on the art to mark the face and try again.', 'code': 'no-target-face'}), 422
        target_face = max(target_faces, key=face_area)

    # Swapping a second time onto the already-swapped face (same landmarks) pushes the
    # result noticeably further toward the source identity; one pass on painted art
    # tends to land somewhere between the painted character and the person.
    result = target_img.copy()
    for _ in range(SWAP_PASSES):
        result = swapper.get(result, target_face, source_face, paste_back=True)
    return jsonify({'image': encode_image(result), 'approximate': approximate})


if __name__ == '__main__':
    port = int(os.environ.get('FACESWAP_PORT', '8787'))
    app.run(host='127.0.0.1', port=port)
