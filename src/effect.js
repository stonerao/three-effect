/**
 * 预览模型
 */
import * as THREE from 'three'
export default {
    effect1() {
        const vertexShader = `
		varying vec3 vPosition;
		varying vec2 vUv;
		void main() { 
			vUv = uv; 
			vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
			gl_Position = projectionMatrix * mvPosition;
		}
		`;
        const fragmentShader = `

        
		uniform float iTime;
		const float PI = 3.14159265359;


        float random(float p){
            return fract(sin(p) * 10000.0);
        } 
        
        float noise(vec2 p){
            float t = iTime / 2000.0;
            if(t > 1.0) t -= floor(t);
            return random(p.x * 14. + p.y * sin(t) * 0.5);
        }

        vec2 sw(vec2 p){
            return vec2(floor(p.x), floor(p.y));
        }
        
        vec2 se(vec2 p){
            return vec2(ceil(p.x), floor(p.y));
        }
        
        vec2 nw(vec2 p){
            return vec2(floor(p.x), ceil(p.y));
        }
        
        vec2 ne(vec2 p){
            return vec2(ceil(p.x), ceil(p.y));
        }

        float smoothNoise(vec2 p){
            vec2 inter = smoothstep(0.0, 1.0, fract(p));
            float s = mix(noise(sw(p)), noise(se(p)), inter.x);
            float n = mix(noise(nw(p)), noise(ne(p)), inter.x);
            return mix(s, n, inter.y);
        }

        mat2 rotate (in float theta){
            float c = cos(theta);
            float s = sin(theta);
            return mat2(c, -s, s, c);
        }

        float circ(vec2 p){
            float r = length(p);
            r = log(sqrt(r));
            return abs(mod(4.0 * r, PI * 2.0) - PI) * 3.0 + 0.2;
        }

        float fbm(in vec2 p){
            float z = 2.0;
            float rz = 0.0;
            vec2 bp = p;
            for(float i = 1.0; i < 6.0; i++){
                rz += abs((smoothNoise(p) - 0.5)* 2.0) / z;
                z *= 2.0;
                p *= 2.0;
            }
            return rz;
        }
        float distanceTo(vec2 src, vec2 dst) {
			float dx = src.x - dst.x;
			float dy = src.y - dst.y;
			float dv = dx * dx + dy * dy;
			return sqrt(dv);
		}
		varying vec2 vUv; 
		uniform vec2 iResolution; 
		void main() { 
            float len = distanceTo(vec2(0.5, 0.5), vec2(vUv.x, vUv.y)) * 2.0; 

            vec2 p = vUv - 0.5;
            p.x *= iResolution.x / iResolution.y;
            p *= 8.0;
            float rz = fbm(p);
            p /= exp(mod(iTime * 2.0, PI));
            rz *= pow(abs(0.1 - circ(p)), 0.9);
            vec3 col = vec3(0.2, 0.1, 0.643); 
            
			gl_FragColor = vec4(col / rz,  1.0 - pow(len, 3.0))  ;
			
		}
		`;
        const geometry = new THREE.PlaneGeometry(20, 20);
        const material = new THREE.ShaderMaterial({
            uniforms: {
                iTime: {
                    value: 0
                },
                iResolution: {
                    value: new THREE.Vector2(1000, 1000)
                }
            },
            side: 2,
            depthWrite: false,
            transparent: true,
            vertexShader: vertexShader,
            fragmentShader: fragmentShader
        })
        const plane = new THREE.Mesh(geometry, material);
        plane.rotation.x = -Math.PI / 2;
        return plane
    },
    effect2() {
        const vertexShader = `
		varying vec3 vPosition;
		varying vec2 vUv;
		void main() { 
			vUv = uv; 
			vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
			gl_Position = projectionMatrix * mvPosition;
		}
		`;
        const fragmentShader = `

		const float PI = 3.14159265359; 
        
		uniform float iTime;
		uniform vec2 iResolution; 
          
		varying vec2 vUv;
        float distanceTo(vec2 src, vec2 dst) {
			float dx = src.x - dst.x;
			float dy = src.y - dst.y;
			float dv = dx * dx + dy * dy;
			return sqrt(dv);
		}

        vec3 hsb2rgb( in vec3 c ){
            vec3 rgb = clamp(abs(mod(c.x*6.0+vec3(0.0,4.0,2.0),
                                     6.0)-3.0)-1.0,
                             0.0,
                             1.0 );
            rgb = rgb*rgb*(3.0-2.0*rgb);
            return c.z * mix( vec3(1.0), rgb, c.y);
        }
        
        vec2 rotate2D (vec2 _st, float _angle) {
            _st =  mat2(cos(_angle),-sin(_angle),
                        sin(_angle),cos(_angle)) * _st;
            return _st;
        }
        
		void main() {  
            float len = distanceTo(vec2(0.5, 0.5), vec2(vUv.x, vUv.y)) * 2.0; 
            vec2 p = (vUv-0.5) * 4.0;
            vec3 color = hsb2rgb(vec3(fract(iTime*.1),1.,1.));
            float r = length(p);
            float w = .3;
            p = rotate2D(p,(r*PI*6.-iTime*2.));
            color *= smoothstep(-w,.0,p.x)*smoothstep(w,.0,p.x);
            color *= abs(1./(sin(pow(r,2.)*2.-iTime*1.3)*6.))*.4;
            
			gl_FragColor = vec4(color,  pow(1.0 - len, 2.0))  ;
			
		}
		`;
        const geometry = new THREE.PlaneGeometry(20, 20);
        const material = new THREE.ShaderMaterial({
            uniforms: {
                iTime: {
                    value: 0
                },
                iResolution: {
                    value: new THREE.Vector2(1000, 1000)
                }
            },
            side: 2,
            depthWrite: false,
            transparent: true,
            vertexShader: vertexShader,
            fragmentShader: fragmentShader
        })
        const plane = new THREE.Mesh(geometry, material);
        plane.rotation.x = -Math.PI / 2;
        return plane
    },
    effect3() {
        const vertexShader = `
		varying vec3 vPosition;
		varying vec2 vUv;
		void main() { 
			vUv = uv; 
			vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
			gl_Position = projectionMatrix * mvPosition;
		}
		`;
        const fragmentShader = `

		const float PI = 3.14159265359; 
        
		uniform float iTime;
		uniform vec2 iResolution; 
          
		varying vec2 vUv;
        
        vec3 firePalette(float i){

            float T = 1400. + 1300.*i; // Temperature range (in Kelvin).
            vec3 L = vec3(7.4, 5.6, 4.4); // Red, green, blue wavelengths (in hundreds of nanometers).
            L = pow(L,vec3(5)) * (exp(1.43876719683e5/(T*L)) - 1.);
            return 1. - exp(-5e8/L); // Exposure level. Set to "50." For "70," change the "5" to a "7," etc.
        } 
        vec3 hash33(vec3 p){ 
            
            float n = sin(dot(p, vec3(7, 157, 113)));    
            return fract(vec3(2097152, 262144, 32768)*n); 
        }
        
        float voronoi(vec3 p){

            vec3 b, r, g = floor(p);
            p = fract(p); // "p -= g;" works on some GPUs, but not all, for some annoying reason.
            
            float d = 1.;  
            for(int j = -1; j <= 1; j++) {
                for(int i = -1; i <= 1; i++) {
                    
                    b = vec3(i, j, -1);
                    r = b - p + hash33(g+b);
                    d = min(d, dot(r,r));
                    
                    b.z = 0.0;
                    r = b - p + hash33(g+b);
                    d = min(d, dot(r,r));
                    
                    b.z = 1.;
                    r = b - p + hash33(g+b);
                    d = min(d, dot(r,r));
                        
                }
            }
            
            return d; // Range: [0, 1]
        }
        
        float noiseLayers(in vec3 p) {
        
            vec3 t = vec3(0., 0., p.z + iTime*1.5);

            const int iter = 5; // Just five layers is enough.
            float tot = 0., sum = 0., amp = 1.; // Total, sum, amplitude.

            for (int i = 0; i < iter; i++) {
                tot += voronoi(p + t) * amp; // Add the layer to the total.
                p *= 2.; // Position multiplied by two.
                t *= 1.5; // Time multiplied by less than two.
                sum += amp; // Sum of amplitudes.
                amp *= .5; // Decrease successive layer amplitude, as normal.
            }
            
            return tot/sum; // Range: [0, 1].
        }
        float distanceTo(vec2 src, vec2 dst) {
			float dx = src.x - dst.x;
			float dy = src.y - dst.y;
			float dv = dx * dx + dy * dy;
			return sqrt(dv);
		}

        
		void main() { 
            float len = distanceTo(vec2(0.5, 0.5), vec2(vUv.x, vUv.y)) * 2.0;  
            vec2 uv = (vUv-0.5) * 2.0;
            
            uv += vec2(sin(iTime*.5)*.25, cos(iTime*.5)*.125);
            
            vec3 rd = normalize(vec3(uv.x, uv.y, 3.1415926535898/8.));
        
            float cs = cos(iTime*.25), si = sin(iTime*.25); 
            rd.xy = rd.xy*mat2(cs, -si, si, cs);  
            float c = noiseLayers(rd*2.);
            
            c = max(c + dot(hash33(rd)*2. - 1., vec3(.015)), 0.);
        
            c *= sqrt(c)*1.5; // Contrast.
            vec3 col = firePalette(c); // Palettization.
            col = mix(col, col.zyx*.15 + c*.85, min(pow(dot(rd.xy, rd.xy)*1.2, 1.5), 1.)); // Color dispersion.
            col = pow(col, vec3(1.25)); // Tweaking the contrast a little.
    
			gl_FragColor = vec4(sqrt(clamp(col, 0., 1.)),  1.0 - pow(len, 2.0));
			
		}
		`;
        const geometry = new THREE.PlaneGeometry(20, 20);
        const material = new THREE.ShaderMaterial({
            uniforms: {
                iTime: {
                    value: 0
                },
                iResolution: {
                    value: new THREE.Vector2(1000, 1000)
                }
            },
            side: 2,
            depthWrite: false,
            transparent: true,
            vertexShader: vertexShader,
            fragmentShader: fragmentShader
        })
        const plane = new THREE.Mesh(geometry, material);
        plane.rotation.x = -Math.PI / 2;
        return plane
    }
}