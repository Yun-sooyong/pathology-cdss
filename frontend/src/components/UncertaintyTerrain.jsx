import { useEffect, useRef } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";

// 실제 픽셀별 MC-Dropout sigma 맵이 아직 없어서, 기권 영역 근처가 높아지도록 설계된
// 시뮬레이션 패턴. 나중에 워커가 픽셀별 sigma 그리드를 내려주면 generateHeights()만
// 실제 데이터로 교체하면 됨 (grid 구조, 좌표계 그대로 유지 가능).
const GRID_SIZE = 32;

function generateHeights(seedX = 0.4, seedY = 0.55) {
  const heights = [];
  for (let y = 0; y < GRID_SIZE; y++) {
    const row = [];
    for (let x = 0; x < GRID_SIZE; x++) {
      const nx = x / GRID_SIZE;
      const ny = y / GRID_SIZE;
      const dx = nx - seedX;
      const dy = ny - seedY;
      const dist = Math.sqrt(dx * dx + dy * dy);
      const peak = Math.exp(-dist * dist * 18);
      const noise = Math.sin(nx * 20) * Math.cos(ny * 17) * 0.05;
      row.push(Math.max(0, peak + noise));
    }
    heights.push(row);
  }
  return heights;
}

export default function UncertaintyTerrain() {
  const containerRef = useRef(null);

  useEffect(() => {
    const container = containerRef.current;
    const width = container.clientWidth;
    const height = 360;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color("#090d16");

    const camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 100);
    camera.position.set(0, 6, 9);

    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(width, height);
    container.appendChild(renderer.domElement);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.maxPolarAngle = Math.PI / 2.1;

    const heights = generateHeights();
    const geometry = new THREE.PlaneGeometry(8, 8, GRID_SIZE - 1, GRID_SIZE - 1);
    geometry.rotateX(-Math.PI / 2);

    const posAttr = geometry.attributes.position;
    const colors = [];
    const low = new THREE.Color("#1f2937");
    const high = new THREE.Color("#f59e0b");

    for (let i = 0; i < posAttr.count; i++) {
      const gx = i % GRID_SIZE;
      const gy = Math.floor(i / GRID_SIZE);
      const h = heights[gy][gx];
      posAttr.setY(i, h * 2.2);
      const c = low.clone().lerp(high, h);
      colors.push(c.r, c.g, c.b);
    }
    geometry.setAttribute("color", new THREE.Float32BufferAttribute(colors, 3));
    geometry.computeVertexNormals();

    const material = new THREE.MeshStandardMaterial({ vertexColors: true, side: THREE.DoubleSide });
    scene.add(new THREE.Mesh(geometry, material));

    const wireGeo = new THREE.WireframeGeometry(geometry);
    const wireMat = new THREE.LineBasicMaterial({ color: "#1f2937", transparent: true, opacity: 0.25 });
    scene.add(new THREE.LineSegments(wireGeo, wireMat));

    scene.add(new THREE.AmbientLight("#ffffff", 0.6));
    const dirLight = new THREE.DirectionalLight("#ffffff", 0.8);
    dirLight.position.set(5, 8, 5);
    scene.add(dirLight);

    let frameId;
    function animate() {
      frameId = requestAnimationFrame(animate);
      controls.update();
      renderer.render(scene, camera);
    }
    animate();

    function handleResize() {
      const w = container.clientWidth;
      camera.aspect = w / height;
      camera.updateProjectionMatrix();
      renderer.setSize(w, height);
    }
    window.addEventListener("resize", handleResize);

    return () => {
      cancelAnimationFrame(frameId);
      window.removeEventListener("resize", handleResize);
      controls.dispose();
      renderer.dispose();
      geometry.dispose();
      material.dispose();
      container.removeChild(renderer.domElement);
    };
  }, []);

  return (
    <div>
      <div
        ref={containerRef}
        className="w-full overflow-hidden rounded-lg border border-border"
        style={{ height: 360 }}
      />
      <div className="mt-2 flex items-center justify-between text-[11px] text-muted-foreground">
        <span>드래그로 회전 · 스크롤로 확대/축소</span>
        <span className="flex items-center gap-1.5">
          <span className="h-2 w-2 rounded-full bg-seg-abstain" /> 높을수록 불확실성 큼 (시뮬레이션 · 실제 σ맵 연동 전)
        </span>
      </div>
    </div>
  );
}