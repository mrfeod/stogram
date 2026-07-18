# Stogram: Stereogram Viewer

<p align="center"><img src="src/favicon/android-chrome-192x192.png" alt="Stogram icon" height="128"></p>

An [autostereogram](https://en.wikipedia.org/wiki/Autostereogram) and depth map viewer: reconstructs a depth map, builds a 3D surface or layers, and allows the texture to be changed independently.

<div align="center" style="display: flex; align-items: center; gap: 10px;">
  <img src="src/images/camel.jpg" alt="Camel stereogram" style="max-height: 512px; max-width: 31%;">
  <span>→</span>
  <img src="src/demo/depth.jpg" alt="Reconstructed camel depth map" style="max-height: 512px; max-width: 31%;">
  <span>→</span>
  <video src="src/demo/example.mp4" controls style="max-height: 512px; max-width: 31%;">
    <a href="src/demo/example.mp4">View the 3D result</a>
  </video>
</div>

## Features

- Depth map reconstruction from a color autostereogram.
- Loading grayscale images as ready-made depth maps.
- Surface, layer, and 2D depth map modes.
- Convex and concave geometry.
- Independent texture replacement without losing the original stereogram and depth.

## Loading Images

A color image is treated as a stereogram and also becomes the texture.

A grayscale image is treated as a ready-made depth map.

A depth map replaces the geometry while preserving the current texture. If there is no texture yet, the map itself is used.

## Interface

| Control | Action |
| --- | --- |
| `＋` | Load a stereogram or depth map |
| `▧` | Replace the texture |
| `↻` | Recalculate the depth map |
| `⊕` | Reset the camera |
| `▶` | Auto-rotation |
| `◠` / `◡` | Convex / concave geometry |
| `❍` | 3D surface |
| `≡` | Layers |
| `◩` | 2D depth map |

| Slider | Action |
| --- | --- |
| `↕︎` | Depth scale |
| `≡` | Number of layers |
| `↔` | Detected stereogram period; adjust manually if needed |

## Controls

- Drag — rotation.
- Mouse wheel or pinch — zoom.
- Double-click — reset the camera.
- Regular drop — load a geometry source.
- Drop onto `▧` or `Shift + drop` — replace the texture.

## Depth Map Calculation

```text
Stereogram
    ↓ original resolution
Global grayscale period search
    ↓ robust upper percentile of unique local repeats
    ↓ candidate periods: background − 30 … background + 5
9×9 local grayscale matching cost
    ↓ subpixel minimum-cost period per pixel
Confidence from the two best matches
    ↓ ambiguous pixels refined from texture-consistent neighbours
3×3 median filter
    ↓ background-period subtraction
Largest foreground component
    ↓ mask-aware Gaussian smoothing
Small-hole filling + MAD spike suppression
    ↓ texture-guided weighted median
    ↓
Ready depth map
    ↓
3D mesh or discrete layers
```
