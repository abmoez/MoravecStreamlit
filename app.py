import streamlit as st
import cv2
import numpy as np

def calcV(window1, window2):
    win1 = np.int32(window1)
    win2 = np.int32(window2)
    diff = win1 - win2
    diff = diff * diff
    return np.sum(diff)


def getWindow(img, i, j, win_size):
    if win_size % 2 == 0:
        return None
    half_size = int(win_size / 2)
    start_x = i - half_size
    start_y = j - half_size
    end_x = i + half_size + 1
    end_y = j + half_size + 1
    return img[start_x:end_x, start_y:end_y]


def getWindowWithRange(img, i, j, win_size):
    if win_size % 2 == 0:
        return None
    half_size = int(win_size / 2)
    start_x = i - half_size
    start_y = j - half_size
    end_x = i + half_size + 1
    end_y = j + half_size + 1
    win = img[start_x:end_x, start_y:end_y]
    return win, start_x, end_x, start_y, end_y


def get8directionWindow(img, i, j, win_size, win_offset):
    half_size = int(win_size / 2)
    win_tl = img[i - win_offset - half_size:i - win_offset + half_size + 1,
                 j - win_offset - half_size:j - win_offset + half_size + 1]
    win_t = img[i - win_offset - half_size:i - win_offset + half_size + 1,
                j - half_size:j + half_size + 1]
    win_tr = img[i - win_offset - half_size:i - win_offset + half_size + 1,
                 j + win_offset - half_size:j + win_offset + half_size + 1]
    win_l = img[i - half_size:i + half_size + 1,
                j - win_offset - half_size:j - win_offset + half_size + 1]
    win_r = img[i - half_size:i + half_size + 1,
                j + win_offset - half_size:j + win_offset + half_size + 1]
    win_bl = img[i + win_offset - half_size:i + win_offset + half_size + 1,
                 j - win_offset - half_size:j - win_offset + half_size + 1]
    win_b = img[i + win_offset - half_size:i + win_offset + half_size + 1,
                j - half_size:j + half_size + 1]
    win_br = img[i + win_offset - half_size:i + win_offset + half_size + 1,
                 j + win_offset - half_size:j + win_offset + half_size + 1]
    return win_tl, win_t, win_tr, win_l, win_r, win_bl, win_b, win_br


def nonMaximumSupression(mat, nonMaxValue=0):
    mask = np.zeros(mat.shape, mat.dtype) + nonMaxValue
    max_value = np.max(mat)
    loc = np.where(mat == max_value)
    row = loc[0]
    col = loc[1]
    mask[row, col] = max_value
    return mask, row, col


def getScore(item):
    return item[2]


def getKeypoints(keymap, nonMaxValue, nFeature=-1):
    loc = np.where(keymap != nonMaxValue)
    xs = loc[1]
    ys = loc[0]
    print(len(xs), 'keypoints were found.')
    kps = []
    for x, y in zip(xs, ys):
        kps.append([x, y, keymap[y, x]])

    if nFeature != -1:
        kps.sort(key=getScore)
        kps = kps[:nFeature]
        print(len(kps), 'keypoints were selected.')
    return kps


def drawKeypoints(img, kps):
    for kp in kps:
        pt = (kp[0], kp[1])
        cv2.circle(img, pt, 3, [0, 0, 255], 1, cv2.LINE_AA)
    return img


def getMoravecKps(img_path, win_size=3, win_offset=1, nonMax_size=5, nonMaxValue=0, nFeature=-1, thCRF=-1):
    print("step 1:read image")
    img_rgb = cv2.imread(img_path)
    img = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    img_h = img.shape[0]
    img_w = img.shape[1]
    print("=>image size:", img_h, '*', img_w)

    keymap = np.zeros([img_h, img_w], np.int32)

    print("step 2:calculate score value using sliding window")
    safe_range = win_offset + win_size
    for i in range(safe_range, img_h - safe_range):
        for j in range(safe_range, img_w - safe_range):
            win = getWindow(img, i, j, win_size)
            win_tl, win_t, win_tr, win_l, win_r, win_bl, win_b, win_br = get8directionWindow(
                img, i, j, win_size, win_offset)
            v1 = calcV(win, win_tl)
            v2 = calcV(win, win_t)
            v3 = calcV(win, win_tr)
            v4 = calcV(win, win_l)
            v5 = calcV(win, win_r)
            v6 = calcV(win, win_bl)
            v7 = calcV(win, win_b)
            v8 = calcV(win, win_br)
            c = min(v1, v2, v3, v4, v5, v6, v7, v8)
            keymap[i, j] = c

    if thCRF == -1:
        mean_c = np.mean(keymap)
        print('=>auto threshold for score value:', mean_c)
    else:
        mean_c = thCRF
        print('=>threshold for score value:', mean_c)

    print("step 3:filter keypoints using threshold...")
    cv2.imwrite("keymap.jpg", keymap)
    keymap = np.where(keymap < mean_c, 0, keymap)
    cv2.imwrite("keymap_th.jpg", keymap)

    print("step 4:non maximum suppression...")
    for i in range(safe_range, img_h - safe_range):
        for j in range(safe_range, img_w - safe_range):
            win, stx, enx, sty, eny = getWindowWithRange(keymap, i, j, nonMax_size)
            nonMax_win, row, col = nonMaximumSupression(win)
            keymap[stx:enx, sty:eny] = nonMax_win
    cv2.imwrite("keymap_nonMax.jpg", keymap)

    print("step 5:get keypoint location and draw points.")
    kps = getKeypoints(keymap, nonMaxValue=nonMaxValue, nFeature=nFeature)
    img_kps = drawKeypoints(img_rgb, kps)
    return kps, img_kps

# ----------------------------
# Streamlit app
# ----------------------------

st.set_page_config(page_title="Moravec Corner Detection", layout="centered")
st.title("🧠 Moravec Keypoint Detector")

# 📊 Section: Explanation image and description
st.markdown("## 📘 About the Moravec Operator")
st.image("moravec_diagram.jpg", caption="Illustration of the Moravec corner detection process", use_container_width=True)

st.markdown("""
The **Moravec Corner Detection** algorithm is one of the earliest corner detection techniques. 
It detects interest points by evaluating how much a small window shifts in intensity in multiple directions. 

The algorithm slides a window across the image and calculates the intensity difference in **8 directions**.
If the **minimum difference** across all directions is **large**, the pixel is likely a corner.

### 📌 Steps:
1. Convert image to grayscale.
2. Slide a small window across the image.
3. Compare each window with shifted windows in 8 directions.
4. Compute the minimum of all shift differences as the corner score.
5. Apply thresholding and **non-maximum suppression** to localize corners.

Moravec’s detector is simple and efficient, making it a good baseline before moving to more advanced detectors like Harris or SIFT.
                        
""")

# 🔗 Section: GitHub Link
st.markdown("🔗 [View Algorithm Source on GitHub](https://github.com/abmoez/MoravecCornerDetection)")

# 📤 Section: Upload Image
st.markdown("---")
st.header("📷 Upload an Image")
uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

if uploaded_file:
    with open("img.jpg", "wb") as f:
        f.write(uploaded_file.read())

    with st.spinner("🧠 Processing with Moravec Operator..."):
        kps, img_kps = getMoravecKps("img.jpg", nFeature=300)
        cv2.imwrite("moravec.jpg", img_kps)

    st.success("✅ Keypoints detected!")

    # ========== Row 1 ==========
    col1, col2 = st.columns(2)

    # First Image: keymap_th
    with col1:
        st.markdown("#### Thresholded Keymap")
        st.image("keymap_th.jpg", use_container_width=True)

    # Second Image: keymap_nonMax
    with col2:
        st.markdown("#### NMS Keymap")
        st.image("keymap_nonMax.jpg", use_container_width=True)

    # ========== Row 2 ==========
    st.markdown("---")
    st.markdown("#### Detected Keypoints on Original Image")
    st.image("moravec.jpg", use_container_width=True)

    

st.markdown("""
    #### 💻 CSE Batch 58:
    1. Abdelmoez Ashraf Abdallah
    2. Omar Ashraf Helmy
    3. Mahmoud Ashraf Gad
    4. Ahmed Hesham Hassan
    5. Abdelrahman Ashraf Mahmoud
                    
    """)
