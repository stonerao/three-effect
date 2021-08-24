/**
 * 预览模型
 */
import * as THREE from 'three'
const vertexShader = `
		varying vec3 vPosition;
		varying vec2 vUv;
		void main() { 
			vUv = uv; 
			vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
			gl_Position = projectionMatrix * mvPosition;
		}
		`;
const getMesh = (fragmentShader) => {
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
    return plane;
}
export default {
    effect1() {
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
        
        return getMesh(fragmentShader);
    },
    effect2() {
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
        return getMesh(fragmentShader);
    },
    effect3() {
        
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
        return getMesh(fragmentShader);
    },
    effect4() {
      
        const fragmentShader = `
        uniform float ratio;

        float PI2 = 6.28318530718;
        float PI = 3.1416;

		uniform float iTime;
		uniform vec2 iResolution; 
          
		varying vec2 vUv;
        float vorocloud(vec2 p){
            float f = 0.0;
            vec2 pp = cos(vec2(p.x * 14.0, (16.0 * p.y + cos(floor(p.x * 30.0)) + iTime * PI2)) );
            p = cos(p * 12.1 + pp * 10.0 + 0.5 * cos(pp.x * 10.0));
            
            vec2 pts[4];
            
            pts[0] = vec2(0.5, 0.6);
            pts[1] = vec2(-0.4, 0.4);
            pts[2] = vec2(0.2, -0.7);
            pts[3] = vec2(-0.3, -0.4);
            
            float d = 5.0;
            
            for(int i = 0; i < 4; i++){
                  pts[i].x += 0.03 * cos(float(i)) + p.x;
                  pts[i].y += 0.03 * sin(float(i)) + p.y;
                d = min(d, distance(pts[i], pp));
            }
            
            f = 2.0 * pow(1.0 - 0.3 * d, 13.0);
            
            f = min(f, 1.0);
            
            return f;
        }
        vec4 scene(vec2 UV){
            float x = UV.x;
            float y = UV.y;
            
            vec2 p = vec2(x, y) - vec2(0.5);
            
            vec4 col = vec4(0.0);
            col.g += 0.02;
            
            float v = vorocloud(p);
            v = 0.2 * floor(v * 5.0);
            
            col.r += 0.1 * v;
            col.g += 0.6 * v;
            col.b += 0.5 * pow(v, 5.0);
            
            
            v = vorocloud(p * 2.0);
            v = 0.2 * floor(v * 5.0);
            
            col.r += 0.1 * v;
            col.g += 0.2 * v;
            col.b += 0.01 * pow(v, 5.0);
            
            col.a = 1.0;
            
            return col;
        }
         
        float distanceTo(vec2 src, vec2 dst) {
			float dx = src.x - dst.x;
			float dy = src.y - dst.y;
			float dv = dx * dx + dy * dy;
			return sqrt(dv);
		}

        
		void main() { 
            float len = distanceTo(vec2(0.5, 0.5), vec2(vUv.x, vUv.y)) * 2.0; 
             

			gl_FragColor = scene(vUv);
			
		}
		`;
        return getMesh(fragmentShader);
    },
    effect5() {
        const fragmentShader = `
        uniform float ratio;

        float M_PI = 3.1415926;
        float M_TWO_PI = 6.28318530718;
        vec3 iMouse = vec3(0.0, 0.0 ,0.0 );
		uniform float iTime;
		uniform vec2 iResolution; 
          
		varying vec2 vUv;
        float rand(vec2 n) {
            return fract(sin(dot(n, vec2(12.9898,12.1414))) * 83758.5453);
        }
        
        float noise(vec2 n) {
            const vec2 d = vec2(0.0, 1.0);
            vec2 b = floor(n);
            vec2 f = smoothstep(vec2(0.0), vec2(1.0), fract(n));
            return mix(mix(rand(b), rand(b + d.yx), f.x), mix(rand(b + d.xy), rand(b + d.yy), f.x), f.y);
        }
        
        vec3 ramp(float t) {
            return t <= .5 ? vec3( 1. - t * 1.4, .2, 1.05 ) / t : vec3( .3 * (1. - t) * 2., .2, 1.05 ) / t;
        }
        vec2 polarMap(vec2 uv, float shift, float inner) {
        
            uv = vec2(0.5) - uv;
            
            
            float px = 1.0 - fract(atan(uv.y, uv.x) / 6.28 + 0.25) + shift;
            float py = (sqrt(uv.x * uv.x + uv.y * uv.y) * (1.0 + inner * 2.0) - inner) * 2.0;
            
            return vec2(px, py);
        }
        float fire(vec2 n) {
            return noise(n) + noise(n * 2.1) * .6 + noise(n * 5.4) * .42;
        }
        
        float shade(vec2 uv, float t) {
            uv.x += uv.y < .5 ? 23.0 + t * .035 : -11.0 + t * .03;    
            uv.y = abs(uv.y - .5);
            uv.x *= 35.0;
            
            float q = fire(uv - t * .013) / 2.0;
            vec2 r = vec2(fire(uv + q / 2.0 + t - uv.x - uv.y), fire(uv + q - t));
            
            return pow((r.y + r.y) * max(.0, uv.y) + .1, 4.0);
        }
        
        vec3 color(float grad) {
            
            float m2 = iMouse.z < 0.0001 ? 1.15 : iMouse.y * 3.0 / iResolution.y;
            grad =sqrt( grad);
            vec3 color = vec3(1.0 / (pow(vec3(0.5, 0.0, .1) + 2.61, vec3(2.0))));
            vec3 color2 = color;
            color = ramp(grad);
            color /= (m2 + max(vec3(0), color));
            
            return color;
        
        }
        
         
        float distanceTo(vec2 src, vec2 dst) {
			float dx = src.x - dst.x;
			float dy = src.y - dst.y;
			float dv = dx * dx + dy * dy;
			return sqrt(dv);
		}

        
		void main() { 
            float m1 = iMouse.z < 0.0001 ? 3.6 : iMouse.x * 5.0 / iResolution.x;
    
            float t = iTime;
            vec2 uv = vUv;
            float ff = 1.0 - uv.y;
            uv.x -= (iResolution.x / iResolution.y - 1.0) / 2.0;
            vec2 uv2 = uv;
            uv2.y = 1.0 - uv2.y;
            uv = polarMap(uv, 1.3, m1);
            uv2 = polarMap(uv2, 1.9, m1);

            vec3 c1 = color(shade(uv, t)) * ff;
            vec3 c2 = color(shade(uv2, t)) * (1.0 - ff);
             

			gl_FragColor = vec4(c1 + c2, 1.0);;
			
		}
		`;
        return getMesh(fragmentShader);
    },
    effect6() {
        const fragmentShader = `
        uniform float ratio;

        float PI = 3.1415926;
		uniform float iTime;
		uniform vec2 iResolution; 
		varying vec2 vUv;
        
		void main() { 
            vec2 p = (vUv - 0.5) * 2.0;
            float tau = PI * 2.0;
            float a = atan(p.x,p.y);
            float r = length(p)*0.75;
            vec2 uv = vec2(a/tau,r);
            
            //get the color
            float xCol = (uv.x - (iTime / 3.0)) * 3.0;
            xCol = mod(xCol, 3.0);
            vec3 horColour = vec3(0.25, 0.25, 0.25);
            
            if (xCol < 1.0) {
                
                horColour.r += 1.0 - xCol;
                horColour.g += xCol;
            }
            else if (xCol < 2.0) {
                
                xCol -= 1.0;
                horColour.g += 1.0 - xCol;
                horColour.b += xCol;
            }
            else {
                
                xCol -= 2.0;
                horColour.b += 1.0 - xCol;
                horColour.r += xCol;
            }

            // draw color beam
            uv = (2.0 * uv) - 1.0;
            float beamWidth = (0.7+0.5*cos(uv.x*10.0*tau*0.15*clamp(floor(5.0 + 10.0*cos(iTime)), 0.0, 10.0))) * abs(1.0 / (30.0 * uv.y));
            vec3 horBeam = vec3(beamWidth); 
			gl_FragColor = vec4((( horBeam) * horColour), 1.0);
			
		}
		`;
        return getMesh(fragmentShader);
    },
    effect7() {
        const fragmentShader = `
        uniform float ratio;

        float PI = 3.1415926;
		uniform float iTime;
		uniform vec2 iResolution; 
		varying vec2 vUv;
        
        vec2 rotate(vec2 p, float rad) {
            mat2 m = mat2(cos(rad), sin(rad), -sin(rad), cos(rad));
            return m * p;
        }
        
        vec2 translate(vec2 p, vec2 diff) {
            return p - diff;
        }
        
        vec2 scale(vec2 p, float r) {
            return p*r;
        }
        
        float circle(float pre, vec2 p, float r1, float r2, float power) {
            float leng = length(p);
            float d = min(abs(leng-r1), abs(leng-r2));
            if (r1<leng && leng<r2) pre /= exp(d)/r2;
            float res = power / d;
            return clamp(pre + res, 0.0, 1.0);
        }
        
        float rectangle(float pre, vec2 p, vec2 half1, vec2 half2, float power) {
            p = abs(p);
            if ((half1.x<p.x || half1.y<p.y) && (p.x<half2.x && p.y<half2.y)) {
                pre = max(0.01, pre);
            }
            float dx1 = (p.y < half1.y) ? abs(half1.x-p.x) : length(p-half1);
            float dx2 = (p.y < half2.y) ? abs(half2.x-p.x) : length(p-half2);
            float dy1 = (p.x < half1.x) ? abs(half1.y-p.y) : length(p-half1);
            float dy2 = (p.x < half2.x) ? abs(half2.y-p.y) : length(p-half2);
            float d = min(min(dx1, dx2), min(dy1, dy2));
            float res = power / d;
            return clamp(pre + res, 0.0, 1.0);
        }
        float radiation(float pre, vec2 p, float r1, float r2, int num, float power) {
            float angle = 2.0*PI/float(num);
            float d = 1e10;
            for(int i=0; i<360; i++) {
                if (i>=num) break;
                float _d = (r1<p.y && p.y<r2) ? 
                    abs(p.x) : 
                    min(length(p-vec2(0.0, r1)), length(p-vec2(0.0, r2)));
                d = min(d, _d);
                p = rotate(p, angle);
            }
            float res = power / d;
            return clamp(pre + res, 0.0, 1.0);
        }
        vec3 calc(vec2 p) {
            float dst = 0.0;
            p = scale(p, sin(PI*iTime/1.0)*0.02+1.1);
            {
                vec2 q = p;
                q = rotate(q, iTime * PI / 6.0);
                dst = circle(dst, q, 0.85, 0.9, 0.006);
                dst = radiation(dst, q, 0.87, 0.88, 36, 0.0008);
            }
            {
                vec2 q = p;
                q = rotate(q, iTime * PI / 6.0);
                const int n = 6;
                float angle = PI / float(n);
                q = rotate(q, floor(atan(q.x, q.y)/angle + 0.5) * angle);
                for(int i=0; i<n; i++) {
                    dst = rectangle(dst, q, vec2(0.85/sqrt(2.0)), vec2(0.85/sqrt(2.0)), 0.0015);
                    q = rotate(q, angle);
                }
            }
            {
                vec2 q = p;
                q = rotate(q, iTime * PI / 6.0);
                const int n = 12;
                q = rotate(q, 2.0*PI/float(n)/2.0);
                float angle = 2.0*PI / float(n);
                for(int i=0; i<n; i++) {
                    dst = circle(dst, q-vec2(0.0, 0.875), 0.001, 0.05, 0.004);
                    dst = circle(dst, q-vec2(0.0, 0.875), 0.001, 0.001, 0.008);
                    q = rotate(q, angle);
                }
            }
            {
                vec2 q = p;
                dst = circle(dst, q, 0.5, 0.55, 0.002);
            }
            {
                vec2 q = p;
                q = rotate(q, -iTime * PI / 6.0);
                const int n = 3;
                float angle = PI / float(n);
                q = rotate(q, floor(atan(q.x, q.y)/angle + 0.5) * angle);
                for(int i=0; i<n; i++) {
                    dst = rectangle(dst, q, vec2(0.36, 0.36), vec2(0.36, 0.36), 0.0015);
                    q = rotate(q, angle);
                }
            }
            {
                vec2 q = p;
                q = rotate(q, -iTime * PI / 6.0);
                const int n = 12;
                q = rotate(q, 2.0*PI/float(n)/2.0);
                float angle = 2.0*PI / float(n);
                for(int i=0; i<n; i++) {
                    dst = circle(dst, q-vec2(0.0, 0.53), 0.001, 0.035, 0.004);
                    dst = circle(dst, q-vec2(0.0, 0.53), 0.001, 0.001, 0.001);
                    q = rotate(q, angle);
                }
            }
            {
                vec2 q = p;
                q = rotate(q, iTime * PI / 6.0);
                dst = radiation(dst, q, 0.25, 0.3, 12, 0.005);
            }
            {
                vec2 q = p;
                q = scale(q, sin(PI*iTime/1.0)*0.04+1.1);
                q = rotate(q, -iTime * PI / 6.0);
                for(float i=0.0; i<6.0; i++) {
                    float r = 0.13-i*0.01;
                    q = translate(q, vec2(0.1, 0.0));
                    dst = circle(dst, q, r, r, 0.002);
                    q = translate(q, -vec2(0.1, 0.0));
                    q = rotate(q, -iTime * PI / 12.0);
                }
                dst = circle(dst, q, 0.04, 0.04, 0.004);
            }
            return pow(dst, 2.5) * vec3(1.0, 0.95, 0.8);
        }
		void main() { 
            vec2 uv = (vUv - 0.5) * 2.0;
			gl_FragColor = vec4(calc(uv), 1.0);;
			
		}
		`;
        return getMesh(fragmentShader);
    },
    effect8() {
        const fragmentShader = `
        #define SMOOTH(r,R) (1.0-smoothstep(R-1.0,R+1.0, r))
        #define RANGE(a,b,x) ( step(a,x)*(1.0-step(b,x)) )
        #define RS(a,b,x) ( smoothstep(a-1.0,a+1.0,x)*(1.0-smoothstep(b-1.0,b+1.0,x)) )
        #define M_PI 3.1415926535897932384626433832795

        #define blue1 vec3(0.74,0.95,1.00)
        #define blue2 vec3(0.87,0.98,1.00)
        #define blue3 vec3(0.35,0.76,0.83)
        #define blue4 vec3(0.953,0.969,0.89)
        #define red   vec3(1.00,0.38,0.227)

        #define MOV(a,b,c,d,t) (vec2(a*cos(t)+b*cos(0.1*(t)), c*sin(t)+d*cos(0.1*(t))))

        uniform float ratio;

        float PI = 3.1415926;
		uniform float iTime;
		uniform vec2 iResolution; 
		varying vec2 vUv;
        
        
        float movingLine(vec2 uv, vec2 center, float radius)
        {
            //angle of the line
            float theta0 = 90.0 * iTime;
            vec2 d = uv - center;
            float r = sqrt( dot( d, d ) );
            if(r<radius)
            {
                //compute the distance to the line theta=theta0
                vec2 p = radius*vec2(cos(theta0*M_PI/180.0),
                                    -sin(theta0*M_PI/180.0));
                float l = length( d - p*clamp( dot(d,p)/dot(p,p), 0.0, 1.0) );
                d = normalize(d);
                //compute gradient based on angle difference to theta0
                float theta = mod(180.0*atan(d.y,d.x)/M_PI+theta0,360.0);
                float gradient = clamp(1.0-theta/90.0,0.0,1.0);
                return SMOOTH(l,1.0)+0.5*gradient;
            }
            else return 0.0;
        }

        float circle(vec2 uv, vec2 center, float radius, float width)
        {
            float r = length(uv - center);
            return SMOOTH(r-width/2.0,radius)-SMOOTH(r+width/2.0,radius);
        }

        float circle2(vec2 uv, vec2 center, float radius, float width, float opening)
        {
            vec2 d = uv - center;
            float r = sqrt( dot( d, d ) );
            d = normalize(d);
            if( abs(d.y) > opening )
                return SMOOTH(r-width/2.0,radius)-SMOOTH(r+width/2.0,radius);
            else
                return 0.0;
        }
        float circle3(vec2 uv, vec2 center, float radius, float width)
        {
            vec2 d = uv - center;
            float r = sqrt( dot( d, d ) );
            d = normalize(d);
            float theta = 180.0*(atan(d.y,d.x)/M_PI);
            return smoothstep(2.0, 2.1, abs(mod(theta+2.0,45.0)-2.0)) *
                mix( 0.5, 1.0, step(45.0, abs(mod(theta, 180.0)-90.0)) ) *
                (SMOOTH(r-width/2.0,radius)-SMOOTH(r+width/2.0,radius));
        }

        float triangles(vec2 uv, vec2 center, float radius)
        {
            vec2 d = uv - center;
            return RS(-8.0, 0.0, d.x-radius) * (1.0-smoothstep( 7.0+d.x-radius,9.0+d.x-radius, abs(d.y)))
                + RS( 0.0, 8.0, d.x+radius) * (1.0-smoothstep( 7.0-d.x-radius,9.0-d.x-radius, abs(d.y)))
                + RS(-8.0, 0.0, d.y-radius) * (1.0-smoothstep( 7.0+d.y-radius,9.0+d.y-radius, abs(d.x)))
                + RS( 0.0, 8.0, d.y+radius) * (1.0-smoothstep( 7.0-d.y-radius,9.0-d.y-radius, abs(d.x)));
        }

        float _cross(vec2 uv, vec2 center, float radius)
        {
            vec2 d = uv - center;
            int x = int(d.x);
            int y = int(d.y);
            float r = sqrt( dot( d, d ) );
            if( (r<radius) && ( (x==y) || (x==-y) ) )
                return 1.0;
            else return 0.0;
        }
        float dots(vec2 uv, vec2 center, float radius)
        {
            vec2 d = uv - center;
            float r = sqrt( dot( d, d ) );
            if( r <= 2.5 )
                return 1.0;
            if( ( r<= radius) && ( (abs(d.y+0.5)<=1.0) && ( mod(d.x+1.0, 50.0) < 2.0 ) ) )
                return 1.0;
            else if ( (abs(d.y+0.5)<=1.0) && ( r >= 50.0 ) && ( r < 115.0 ) )
                return 0.5;
            else
                return 0.0;
        }
        float bip1(vec2 uv, vec2 center)
        {
            return SMOOTH(length(uv - center),3.0);
        }
        float bip2(vec2 uv, vec2 center)
        {
            float r = length(uv - center);
            float R = 8.0+mod(87.0*iTime, 80.0);
            return (0.5-0.5*cos(30.0*iTime)) * SMOOTH(r,5.0)
                + SMOOTH(6.0,r)-SMOOTH(8.0,r)
                + smoothstep(max(8.0,R-20.0),R,r)-SMOOTH(R,r);
        }
		void main() { 
            vec2 _uv = vec2(vUv.x * iResolution.x, vUv.y * iResolution.y);
            vec3 finalColor;
            vec2 uv = _uv;
            //center of the image
            vec2 c = vec2(iResolution.x / 2.0, iResolution.y / 2.0);
            finalColor = vec3( 0.3*_cross(uv, c, 240.0) );
            finalColor += ( circle(uv, c, 100.0, 1.0)
                        + circle(uv, c, 165.0, 1.0) ) * blue1;
            finalColor += (circle(uv, c, 240.0, 2.0) );//+ dots(uv,c,240.0)) * blue4;
            finalColor += circle3(uv, c, 313.0, 4.0) * blue1;
            finalColor += triangles(uv, c, 315.0 + 30.0*sin(iTime)) * blue2;
            finalColor += movingLine(uv, c, 240.0) * blue3;
            finalColor += circle(uv, c, 10.0, 1.0) * blue3;
            finalColor += 0.7 * circle2(uv, c, 262.0, 1.0, 0.5+0.2*cos(iTime)) * blue3;
            if( length(uv-c) < 240.0 )
            {
                //animate some bips with random movements
                vec2 p = 130.0*MOV(1.3,1.0,1.0,1.4,3.0+0.1*iTime);
                finalColor += bip1(uv, c+p) * vec3(1,1,1);
                p = 130.0*MOV(0.9,-1.1,1.7,0.8,-2.0+sin(0.1*iTime)+0.15*iTime);
                finalColor += bip1(uv, c+p) * vec3(1,1,1);
                p = 50.0*MOV(1.54,1.7,1.37,1.8,sin(0.1*iTime+7.0)+0.2*iTime);
                finalColor += bip2(uv,c+p) * red;
            }

			gl_FragColor = vec4( finalColor, 1.0 );
			
		}
		`;
        return getMesh(fragmentShader);
    },
    effect9() {
        const fragmentShader = `
        uniform float ratio;

        float PI = 3.1415926;
		uniform float iTime;
		uniform vec2 iResolution; 
		varying vec2 vUv;
        
        const float cloudscale = 1.1;
        const float speed = 0.03;
        const float clouddark = 0.5;
        const float cloudlight = 0.3;
        const float cloudcover = 0.2;
        const float cloudalpha = 8.0;
        const float skytint = 0.5;
        const vec3 skycolour1 = vec3(0.2, 0.4, 0.6);
        const vec3 skycolour2 = vec3(0.4, 0.7, 1.0);

        const mat2 m = mat2( 1.6,  1.2, -1.2,  1.6 );

        vec2 hash( vec2 p ) {
            p = vec2(dot(p,vec2(127.1,311.7)), dot(p,vec2(269.5,183.3)));
            return -1.0 + 2.0*fract(sin(p)*43758.5453123);
        }
        float noise( in vec2 p ) {
            const float K1 = 0.366025404; // (sqrt(3)-1)/2;
            const float K2 = 0.211324865; // (3-sqrt(3))/6;
            vec2 i = floor(p + (p.x+p.y)*K1);	
            vec2 a = p - i + (i.x+i.y)*K2;
            vec2 o = (a.x>a.y) ? vec2(1.0,0.0) : vec2(0.0,1.0); //vec2 of = 0.5 + 0.5*vec2(sign(a.x-a.y), sign(a.y-a.x));
            vec2 b = a - o + K2;
            vec2 c = a - 1.0 + 2.0*K2;
            vec3 h = max(0.5-vec3(dot(a,a), dot(b,b), dot(c,c) ), 0.0 );
            vec3 n = h*h*h*h*vec3( dot(a,hash(i+0.0)), dot(b,hash(i+o)), dot(c,hash(i+1.0)));
            return dot(n, vec3(70.0));	
        }
        
        float fbm(vec2 n) {
            float total = 0.0, amplitude = 0.1;
            for (int i = 0; i < 7; i++) {
                total += noise(n) * amplitude;
                n = m * n;
                amplitude *= 0.4;
            }
            return total;
        }
        
		void main() { 
            vec2 p = (vUv - 0.5) * 2.0;
            vec2 uv = p*vec2(iResolution.x/iResolution.y,1.0);    
            float time = iTime * speed;
            float q = fbm(uv * cloudscale * 0.5);
            
            //ridged noise shape
            float r = 0.0;
            uv *= cloudscale;
            uv -= q - time;
            float weight = 0.8;
            for (int i=0; i<8; i++){
                r += abs(weight*noise( uv ));
                uv = m*uv + time;
                weight *= 0.7;
            }
            
            //noise shape
            float f = 0.0;
            uv = p*vec2(iResolution.x/iResolution.y,1.0);
            uv *= cloudscale;
            uv -= q - time;
            weight = 0.7;
            for (int i=0; i<8; i++){
                f += weight*noise( uv );
                uv = m*uv + time;
                weight *= 0.6;
            }
            
            f *= r + f;
            
            //noise colour
            float c = 0.0;
            time = iTime * speed * 2.0;
            uv = p*vec2(iResolution.x/iResolution.y,1.0);
            uv *= cloudscale*2.0;
            uv -= q - time;
            weight = 0.4;
            for (int i=0; i<7; i++){
                c += weight*noise( uv );
                uv = m*uv + time;
                weight *= 0.6;
            }
            
            //noise ridge colour
            float c1 = 0.0;
            time = iTime * speed * 3.0;
            uv = p*vec2(iResolution.x/iResolution.y,1.0);
            uv *= cloudscale*3.0;
            uv -= q - time;
            weight = 0.4;
            for (int i=0; i<7; i++){
                c1 += abs(weight*noise( uv ));
                uv = m*uv + time;
                weight *= 0.6;
            }
            
            c += c1;
            
            vec3 skycolour = mix(skycolour2, skycolour1, p.y);
            vec3 cloudcolour = vec3(1.1, 1.1, 0.9) * clamp((clouddark + cloudlight*c), 0.0, 1.0);
        
            f = cloudcover + cloudalpha*f*r;
            
            vec3 result = mix(skycolour, clamp(skytint * skycolour + cloudcolour, 0.0, 1.0), clamp(f + c, 0.0, 1.0));

			gl_FragColor = vec4( result, 1.0 );
			
		}
		`;
        return getMesh(fragmentShader);
    }
}