import * as THREE from "https://unpkg.com/three@0.160.0/build/three.module.js";
import { OrbitControls } from "https://unpkg.com/three@0.160.0/examples/jsm/controls/OrbitControls.js";

const canvas = document.getElementById("canvas");
const fileInput = document.getElementById("file");
const playBtn = document.getElementById("play");
const pauseBtn = document.getElementById("pause");
const speedSlider = document.getElementById("speed");
const speedLabel = document.getElementById("speedLabel");

let specimen = null;
let playing = false;
let frameIdx = 0;
let accumulator = 0;

const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
renderer.setSize(window.innerWidth, window.innerHeight);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0xf6f6f6);

const camera = new THREE.PerspectiveCamera(55, window.innerWidth / window.innerHeight, 0.01, 100);
camera.position.set(1.6, 1.2, 1.2);

const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(0, 0, 0.4);
controls.update();

const hemi = new THREE.HemisphereLight(0xffffff, 0x444444, 1.0);
scene.add(hemi);

const dir = new THREE.DirectionalLight(0xffffff, 0.8);
dir.position.set(3, 4, 2);
scene.add(dir);

const groundGeo = new THREE.PlaneGeometry(20, 20);
const groundMat = new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.95, metalness: 0.0 });
const ground = new THREE.Mesh(groundGeo, groundMat);
ground.rotation.x = -Math.PI / 2;
ground.position.y = 0;
scene.add(ground);

const axes = new THREE.AxesHelper(0.5);
axes.position.set(0, 0, 0.01);
scene.add(axes);

let baseMesh = null;
let linkMeshes = [];

function clearCreature() {
  if (baseMesh) scene.remove(baseMesh);
  for (const m of linkMeshes) scene.remove(m);
  baseMesh = null;
  linkMeshes = [];
}

function ensureMeshes(frame) {
  if (!frame) return;
  if (!baseMesh) {
    const g = new THREE.SphereGeometry(0.16, 24, 16);
    const m = new THREE.MeshStandardMaterial({ color: 0xcc8844, roughness: 0.75 });
    baseMesh = new THREE.Mesh(g, m);
    scene.add(baseMesh);
  }
  const n = frame.links.length;
  if (linkMeshes.length !== n) {
    for (const m of linkMeshes) scene.remove(m);
    linkMeshes = [];
    for (let i = 0; i < n; i++) {
      // Use capsules as a generic look; you can later store real geometry per creature in JSON.
      const g = new THREE.CapsuleGeometry(0.05, 0.22, 8, 12);
      const m = new THREE.MeshStandardMaterial({ color: 0x6699cc, roughness: 0.7 });
      const mesh = new THREE.Mesh(g, m);
      scene.add(mesh);
      linkMeshes.push(mesh);
    }
  }
}

function applyFrame(frame) {
  ensureMeshes(frame);
  if (!frame) return;

  const b = frame.base;
  baseMesh.position.set(b.pos[0], b.pos[1], b.pos[2]);
  baseMesh.quaternion.set(b.orn[0], b.orn[1], b.orn[2], b.orn[3]);

  for (let i = 0; i < frame.links.length; i++) {
    const L = frame.links[i];
    const mesh = linkMeshes[i];
    mesh.position.set(L.pos[0], L.pos[1], L.pos[2]);
    mesh.quaternion.set(L.orn[0], L.orn[1], L.orn[2], L.orn[3]);
  }
}

fileInput.addEventListener("change", async (e) => {
  const f = e.target.files?.[0];
  if (!f) return;
  const text = await f.text();
  specimen = JSON.parse(text);
  frameIdx = 0;
  accumulator = 0;
  playing = false;
  clearCreature();
  const first = specimen.trajectory?.[0];
  applyFrame(first);
});

playBtn.addEventListener("click", () => { if (specimen) playing = true; });
pauseBtn.addEventListener("click", () => { playing = false; });

speedSlider.addEventListener("input", () => {
  speedLabel.textContent = `${Number(speedSlider.value).toFixed(1)}x`;
});

function animate(ts) {
  requestAnimationFrame(animate);

  const dt = 1 / 60;
  const speed = Number(speedSlider.value || 1.0);

  if (specimen && playing) {
    const timeStep = specimen.meta?.time_step || (1 / 240);
    accumulator += dt * speed;

    while (accumulator >= timeStep) {
      accumulator -= timeStep;
      frameIdx = (frameIdx + 1) % (specimen.trajectory.length || 1);
      applyFrame(specimen.trajectory[frameIdx]);
    }
  }

  controls.update();
  renderer.render(scene, camera);
}
requestAnimationFrame(animate);

window.addEventListener("resize", () => {
  renderer.setSize(window.innerWidth, window.innerHeight);
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
});