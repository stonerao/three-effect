import './style.css'
import * as THREE from 'three'
import {
    OrbitControls
} from 'three/examples/jsm/controls/OrbitControls.js'
import {
    GLTFLoader
} from 'three/examples/jsm/loaders/GLTFLoader.js'
import * as dat from 'dat.gui'
import Effect from './effect.js'
/**
 * Base
 */

// Debug
const gui = new dat.GUI();


// Canvas
const canvas = document.querySelector('canvas.webgl')

// Scene
const scene = new THREE.Scene()

/**
 * Loaders
 */
const gltfLoader = new GLTFLoader()
const cubeTextureLoader = new THREE.CubeTextureLoader()
const textureLoader = new THREE.TextureLoader()

window.iChannel0 = textureLoader.load('./img.png')

window.iChannel0 .wrapS = THREE.RepeatWrapping;
window.iChannel0 .wrapT = THREE.RepeatWrapping;
window.iChannel0 .repeat.set( 10, 10 );
/**
 * Lights
 */
const directionalLight = new THREE.DirectionalLight('#ffffff', 3)
directionalLight.castShadow = true
directionalLight.shadow.mapSize.set(1024, 1024)
directionalLight.shadow.camera.far = 15
directionalLight.shadow.normalBias = 0.05
directionalLight.position.set(0.25, 3, -2.25)
scene.add(directionalLight)

/**
 * Sizes
 */
const sizes = {
    width: window.innerWidth,
    height: window.innerHeight
}

window.addEventListener('resize', () => {
    // Update sizes
    sizes.width = window.innerWidth
    sizes.height = window.innerHeight

    // Update camera
    camera.aspect = sizes.width / sizes.height
    camera.updateProjectionMatrix()

    // Update renderer
    renderer.setSize(sizes.width, sizes.height)
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
})

/**
 * Camera
 */
// Base camera
const camera = new THREE.PerspectiveCamera(45, sizes.width / sizes.height, 0.1, 10000)
camera.position.set(0, 30, 0)
scene.add(camera)

// Controls
const controls = new OrbitControls(camera, canvas)
controls.enableDamping = true

/**
 * Renderer
 */
const renderer = new THREE.WebGLRenderer({
    canvas: canvas,
    antialias: true
})
renderer.shadowMap.enabled = true
renderer.shadowMap.type = THREE.PCFShadowMap
renderer.physicallyCorrectLights = true
renderer.outputEncoding = THREE.sRGBEncoding
renderer.toneMapping = THREE.ReinhardToneMapping
renderer.toneMappingExposure = 1.5
renderer.setSize(sizes.width, sizes.height)
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))

let effectGroup = null;
const updateEffect = (mesh) => {
    if (effectGroup) {
        scene.remove(effectGroup);
        effectGroup = null;
    }

    effectGroup = mesh;
    scene.add(mesh);
}
const effectList = {};
let last = '';
Object.keys(Effect).forEach((e, i) => {
    const index = (i + 1);
    const name = 'effect' + index;
    effectList[name] = function () {
        updateEffect(Effect[name]());
    }
    gui.add(effectList, name);
    last = name;
})
  
updateEffect(Effect[last]());


/**
 * Animate
 */
const clock = new THREE.Clock()

const tick = () => {
    const elapsedTime = clock.getElapsedTime()

    // Update passes
    scene.traverse((child) => {
        if (child.material) child.material.uniforms.iTime.value = elapsedTime
    })

    // Update controls
    controls.update()


    // Render
    renderer.render(scene, camera)
    // Call tick again on the next frame
    window.requestAnimationFrame(tick)
}

tick()