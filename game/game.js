/**
 * game.js — Plane Shooter (fixed, clean rewrite)
 * Controls: ←/→ or A/D. Space = start/retry. S = shop.
 */

const canvas = document.getElementById('gameCanvas');
const ctx    = canvas.getContext('2d');
const W = canvas.width  = 800;
const H = canvas.height = 600;

let keys = {}, gameState = 'menu', diffMode = 'easy';
let score, lives, frame, spawnTimer;
let player, enemies, pBullets, eBullets, explosions, stars;

const DIFF = {
    easy:   { val:0.0, hp:1, spawn:200, max:3, spd:1.0, col:'#4CAF50', label:'Easy'   },
    medium: { val:0.5, hp:2, spawn:130, max:5, spd:1.5, col:'#FF9800', label:'Medium' },
    hard:   { val:1.0, hp:3, spawn:80,  max:8, spd:2.2, col:'#F44336', label:'Hard'   },
};

// ── Filter helper (module-level, NOT inside gameLoop) ─────────────────────────
function filterArr(arr, predFn) {
    let w = 0;
    for (let i = 0; i < arr.length; i++) if (predFn(arr[i])) arr[w++] = arr[i];
    arr.length = w;
}

// ── Bullet ────────────────────────────────────────────────────────────────────
class Bullet {
    constructor(x, y, dx, dy, col, dmg, maxF, laser) {
        this.x=x; this.y=y; this.dx=dx; this.dy=dy; this.col=col;
        this.dmg=dmg||1; this.maxF=maxF||9999; this.f=0; this.laser=!!laser; this.dead=false;
    }
    update() {
        this.x+=this.dx; this.y+=this.dy; this.f++;
        if (this.y<-10||this.y>H+10||this.f>=this.maxF) this.dead=true;
    }
    draw() {
        ctx.shadowColor=this.col; ctx.shadowBlur=8;
        if (this.laser) {
            ctx.strokeStyle=this.col; ctx.lineWidth=3;
            ctx.beginPath(); ctx.moveTo(this.x,this.y); ctx.lineTo(this.x,this.y+24); ctx.stroke();
        } else {
            ctx.fillStyle=this.col;
            ctx.beginPath(); ctx.ellipse(this.x,this.y,3,6,0,0,Math.PI*2); ctx.fill();
        }
        ctx.shadowBlur=0;
    }
}

// ── Player ────────────────────────────────────────────────────────────────────
class Player {
    constructor() {
        const p=store.getPlane();
        this.x=W/2; this.y=H-80; this.w=p.width; this.h=46;
        this.spd=p.speed; this.col=p.col; this.cd=0; this.inv=0;
    }
    update() {
        if (keys['ArrowLeft']||keys['a']||keys['A']) this.x-=this.spd;
        if (keys['ArrowRight']||keys['d']||keys['D']) this.x+=this.spd;
        this.x=Math.max(this.w/2, Math.min(W-this.w/2, this.x));
        if (this.cd>0) this.cd--;
        if (this.inv>0) this.inv--;
        if (this.cd===0) this._fire();
    }
    _fire() {
        const w=store.getWeapon();
        const mf=w.isLaser ? Math.ceil(H*w.range/w.speed) : 9999;
        if (w.spread>1) {
            const half=Math.floor(w.spread/2);
            for (let i=-half;i<=half;i++)
                pBullets.push(new Bullet(this.x+i*8, this.y-20, i*0.5, -w.speed, w.color, w.damage, mf, !!w.isLaser));
        } else {
            pBullets.push(new Bullet(this.x-9, this.y-20, 0,-w.speed, w.color, w.damage, mf, !!w.isLaser));
            pBullets.push(new Bullet(this.x+9, this.y-20, 0,-w.speed, w.color, w.damage, mf, !!w.isLaser));
        }
        this.cd=w.fireRate;
    }
    draw() {
        if (this.inv>0 && Math.floor(this.inv/4)%2) return;
        const x=this.x, y=this.y, c=this.col;
        ctx.fillStyle=c;
        ctx.beginPath(); ctx.moveTo(x,y-23); ctx.lineTo(x-this.w/2,y+20); ctx.lineTo(x+this.w/2,y+20); ctx.closePath(); ctx.fill();
        ctx.globalAlpha=0.7;
        ctx.beginPath(); ctx.moveTo(x-this.w/2,y+4); ctx.lineTo(x-this.w/2-16,y+18); ctx.lineTo(x-this.w/2+8,y+18); ctx.closePath(); ctx.fill();
        ctx.beginPath(); ctx.moveTo(x+this.w/2,y+4); ctx.lineTo(x+this.w/2+16,y+18); ctx.lineTo(x+this.w/2-8,y+18); ctx.closePath(); ctx.fill();
        ctx.globalAlpha=1;
        ctx.fillStyle=`rgba(255,${100+Math.random()*100|0},0,0.9)`;
        ctx.beginPath(); ctx.moveTo(x-7,y+20); ctx.lineTo(x+7,y+20); ctx.lineTo(x,y+30+Math.random()*8); ctx.closePath(); ctx.fill();
        ctx.fillStyle='#E3F2FD';
        ctx.beginPath(); ctx.ellipse(x,y-4,6,11,0,0,Math.PI*2); ctx.fill();
    }
}

// ── Enemy ─────────────────────────────────────────────────────────────────────
class Enemy {
    constructor(cfg) {
        this.x=Math.random()*(W-60)+30; this.y=-30;
        this.w=36; this.h=40; this.cfg=cfg; this.hp=cfg.hp;
        this.spd=cfg.spd; this.col=cfg.col; this.dv=cfg.val;
        this.scd=60+Math.random()*60|0; this.agg=0.1; this.itimer=0; this.dx=0;
    }
    infer(px,py) {
        const rx=(this.x-px)/W;
        const ry=1-Math.abs(this.y-py)/H;
        let bn=0,brx=0;
        for (let i=0;i<pBullets.length;i++) {
            const b=pBullets[i];
            if (Math.hypot(b.x-this.x,b.y-this.y)<90){bn=1;brx=(b.x-this.x)/W;break;}
        }
        try { this.agg=model.predict({rel_x:rx,rel_y:ry,bullet_near:bn,bullet_rel_x:brx,difficulty:this.dv}); }
        catch(e){ this.agg=this.dv; }
        if (bn&&this.agg>0.4) this.dx=brx<0?1.5:-1.5;
        else if (this.agg>0.55) this.dx=rx<0?-1.2:1.2;
        else if (Math.random()<0.15) this.dx=(Math.random()-0.5)*2;
    }
    update(px,py) {
        this.y+=this.spd;
        this.x=Math.max(this.w/2,Math.min(W-this.w/2,this.x+this.dx));
        this.itimer++;
        if (this.itimer>=20){this.infer(px,py);this.itimer=0;}
        this.scd--;
        if (this.scd<=0){
            if (Math.random()<0.3+this.agg*0.5)
                eBullets.push(new Bullet(this.x,this.y+22,(Math.random()-0.5)*2,5+this.agg*3,this.col));
            this.scd=Math.max(30,90-this.agg*60|0);
        }
    }
    draw() {
        const x=this.x,y=this.y;
        ctx.fillStyle=this.col;
        ctx.beginPath(); ctx.moveTo(x,y+20); ctx.lineTo(x-18,y-18); ctx.lineTo(x+18,y-18); ctx.closePath(); ctx.fill();
        ctx.globalAlpha=0.7;
        ctx.beginPath(); ctx.moveTo(x-18,y-2); ctx.lineTo(x-34,y+12); ctx.lineTo(x-10,y+12); ctx.closePath(); ctx.fill();
        ctx.beginPath(); ctx.moveTo(x+18,y-2); ctx.lineTo(x+34,y+12); ctx.lineTo(x+10,y+12); ctx.closePath(); ctx.fill();
        ctx.globalAlpha=1;
        ctx.fillStyle='#333'; ctx.fillRect(x-16,y-24,32,4);
        ctx.fillStyle=this.col; ctx.fillRect(x-16,y-24,32*(this.hp/this.cfg.hp),4);
        ctx.fillStyle=`hsl(${120-this.agg*120},100%,55%)`;
        ctx.beginPath(); ctx.arc(x,y-30,3,0,Math.PI*2); ctx.fill();
    }
}

// ── Explosion ─────────────────────────────────────────────────────────────────
class Explosion {
    constructor(x,y,col){this.x=x;this.y=y;this.col=col;this.r=2;this.dead=false;}
    update(){this.r+=2.5;if(this.r>=40)this.dead=true;}
    draw(){
        const a=1-this.r/40;
        ctx.strokeStyle=this.col; ctx.lineWidth=2; ctx.globalAlpha=a;
        ctx.beginPath(); ctx.arc(this.x,this.y,this.r,0,Math.PI*2); ctx.stroke();
        ctx.beginPath(); ctx.arc(this.x,this.y,this.r*0.5,0,Math.PI*2); ctx.stroke();
        ctx.globalAlpha=1;
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────
function hit(ax,ay,aw,ah,bx,by,bw,bh){
    return Math.abs(ax-bx)<(aw+bw)/2 && Math.abs(ay-by)<(ah+bh)/2;
}
function initStars(){
    stars=[];
    for(let i=0;i<120;i++)
        stars.push({x:Math.random()*W,y:Math.random()*H,r:Math.random()*1.4+0.2,spd:Math.random()*0.3+0.1});
}
function drawStars(){
    for(const s of stars){
        s.y+=s.spd; if(s.y>H){s.y=0;s.x=Math.random()*W;}
        ctx.fillStyle=`rgba(255,255,255,${0.3+s.r*0.2})`;
        ctx.beginPath(); ctx.arc(s.x,s.y,s.r,0,Math.PI*2); ctx.fill();
    }
}
function resetGame(){
    const p=store.getPlane();
    score=0; lives=p.lives; frame=0; spawnTimer=0;
    player=new Player();
    enemies=[]; pBullets=[]; eBullets=[]; explosions=[];
    gameState='playing';
}

// ── HUD ───────────────────────────────────────────────────────────────────────
function drawHUD(){
    const w=store.getWeapon(), p=store.getPlane();
    ctx.textAlign='left';
    ctx.fillStyle='#fff'; ctx.font='bold 16px monospace';
    ctx.fillText(`Score: ${score}`,14,26);
    ctx.fillStyle='#FFD700'; ctx.font='13px monospace';
    ctx.fillText(`\u{1F4B0} ${store.state.points} pts`,14,44);
    for(let i=0;i<lives;i++){
        ctx.fillStyle=p.col;
        ctx.beginPath(); ctx.moveTo(W-24-i*22,16); ctx.lineTo(W-34-i*22,28); ctx.lineTo(W-14-i*22,28); ctx.closePath(); ctx.fill();
    }
    ctx.fillStyle=DIFF[diffMode].col; ctx.font='bold 13px monospace'; ctx.textAlign='center';
    ctx.fillText(`[ ${DIFF[diffMode].label} ]`,W/2,24);
    ctx.fillStyle=w.color; ctx.font='11px monospace'; ctx.textAlign='right';
    ctx.fillText(`${w.icon} ${w.name}`,W-14,44);
    if(enemies.length>0){
        let ma=0; for(const e of enemies) ma=Math.max(ma,e.agg||0);
        ctx.fillStyle='#333'; ctx.fillRect(14,H-20,100,8);
        ctx.fillStyle=`hsl(${120-ma*120},100%,50%)`; ctx.fillRect(14,H-20,ma*100,8);
        ctx.fillStyle='#888'; ctx.font='10px monospace'; ctx.textAlign='left';
        ctx.fillText(`AI: ${(ma*100).toFixed(0)}%`,14,H-24);
    }
    ctx.textAlign='left';
}

// ── Screens ───────────────────────────────────────────────────────────────────
function drawMenu(){
    ctx.fillStyle='rgba(0,0,20,0.9)'; ctx.fillRect(0,0,W,H);
    ctx.textAlign='center';
    ctx.fillStyle='#00E5FF'; ctx.font='bold 40px monospace'; ctx.fillText('\u2708 PLANE SHOOTER',W/2,138);
    ctx.fillStyle='#7986CB'; ctx.font='14px monospace'; ctx.fillText('ML-Powered Enemy AI  (C++ MLP \u2192 ONNX \u2192 JS)',W/2,166);
    ['easy','medium','hard'].forEach((d,i)=>{
        const bx=W/2-130+i*130,by=196,bw=100,bh=36,cfg=DIFF[d];
        ctx.fillStyle=diffMode===d?cfg.col:'#1a1a3a';
        ctx.strokeStyle=cfg.col; ctx.lineWidth=2;
        ctx.beginPath(); ctx.roundRect(bx-50,by,bw,bh,6); ctx.fill(); ctx.stroke();
        ctx.fillStyle=diffMode===d?'#000':cfg.col; ctx.font='bold 14px monospace';
        ctx.fillText(cfg.label,bx,by+24);
    });
    ctx.fillStyle='#FFD700'; ctx.font='bold 15px monospace';
    ctx.fillText(`\u{1F4B0} ${store.state.points} pts  |  Score: ${store.state.score}`,W/2,272);
    const wn=store.getWeapon(),pn=store.getPlane();
    ctx.fillStyle='#aaa'; ctx.font='13px monospace';
    ctx.fillText(`${pn.icon} ${pn.name}  \u00b7  ${wn.icon} ${wn.name}`,W/2,296);
    ctx.fillStyle='#00E5FF'; ctx.font='bold 20px monospace'; ctx.fillText('[ SPACE ] Start',W/2,345);
    ctx.fillStyle='#FFD700'; ctx.font='bold 18px monospace'; ctx.fillText('[ S ] Armory Shop',W/2,375);

    // Save/Load Buttons on Menu
    const bx1 = W/2 - 130, bx2 = W/2 + 20, by = 410, bw = 110, bh = 30;
    ctx.strokeStyle = '#555'; ctx.lineWidth = 1;
    
    // Save Button
    ctx.fillStyle = '#1a1a3a'; ctx.beginPath(); ctx.roundRect(bx1, by, bw, bh, 4); ctx.fill(); ctx.stroke();
    ctx.fillStyle = '#fff'; ctx.font = 'bold 12px monospace'; ctx.fillText('\uD83D\uDCE5 EXPORT', bx1 + bw/2, by + 20);

    // Load Button
    ctx.fillStyle = '#1a1a3a'; ctx.beginPath(); ctx.roundRect(bx2, by, bw, bh, 4); ctx.fill(); ctx.stroke();
    ctx.fillStyle = '#fff'; ctx.font = 'bold 12px monospace'; ctx.fillText('\uD83D\uDCE4 IMPORT', bx2 + bw/2, by + 20);

    ctx.fillStyle='#555'; ctx.font='12px monospace'; ctx.fillText('Move: \u2190 \u2192 or A/D  |  Auto-fire',W/2,470);
    ctx.textAlign='left';
}
function drawGameOver(){
    ctx.fillStyle='rgba(0,0,0,0.82)'; ctx.fillRect(0,0,W,H);
    ctx.textAlign='center';
    ctx.fillStyle='#F44336'; ctx.font='bold 50px monospace'; ctx.fillText('GAME OVER',W/2,H/2-70);
    ctx.fillStyle='#fff'; ctx.font='22px monospace';
    ctx.fillText(`Score: ${score}  (+${Math.floor(score/10)} pts)`,W/2,H/2-20);
    ctx.fillStyle='#FFD700'; ctx.font='18px monospace';
    ctx.fillText(`Total Points: ${store.state.points}`,W/2,H/2+14);
    ctx.fillStyle='#aaa'; ctx.font='15px monospace';
    ctx.fillText('[ SPACE ] Menu  |  [ S ] Armory',W/2,H/2+60);
    ctx.textAlign='left';
}

// ── Main Loop ─────────────────────────────────────────────────────────────────
function gameLoop(){
    // ALWAYS schedule next frame first — ensures loop never dies on error
    requestAnimationFrame(gameLoop);

    ctx.fillStyle='#050514'; ctx.fillRect(0,0,W,H);
    drawStars();

    if(gameState==='menu')     { drawMenu();           return; }
    if(gameState==='shop')     { drawShop(ctx,W,H);    return; }
    if(gameState==='gameover') { drawGameOver();        return; }

    try {
        frame++; spawnTimer++;
        const cfg=DIFF[diffMode];

        // ── 1. Cleanup dead entities FIRST (so spawn check is accurate) ──
        filterArr(pBullets,  b=>!b.dead);
        filterArr(eBullets,  b=>!b.dead);
        filterArr(explosions,e=>!e.dead);
        filterArr(enemies,   e=>e.hp>0);

        // ── 2. Spawn ──────────────────────────────────────────────────────
        if(spawnTimer>=cfg.spawn && enemies.length<cfg.max){
            enemies.push(new Enemy(cfg));
            spawnTimer=0;
        }

        // ── 3. Update & draw player ───────────────────────────────────────
        player.update(); player.draw();

        // ── 4. Player bullets ─────────────────────────────────────────────
        for(const b of pBullets){ b.update(); b.draw(); }

        // ── 5. Enemies ────────────────────────────────────────────────────
        for(const e of enemies){
            e.update(player.x,player.y); e.draw();

            // Reached bottom
            if(e.y>H+20 && e.hp>0){
                explosions.push(new Explosion(e.x,H-30,e.col));
                e.hp=0;
                if(player.inv===0){ lives--; player.inv=90; }
            }

            // Bullet hits
            for(const b of pBullets){
                if(b.dead||e.hp<=0) continue;
                if(hit(b.x,b.y,5,12,e.x,e.y,e.w,e.h)){
                    b.dead=true;
                    e.hp-=b.dmg;
                    explosions.push(new Explosion(e.x+(Math.random()-.5)*20,e.y+(Math.random()-.5)*10,'#FFF9C4'));
                    if(e.hp<=0){
                        const pts=cfg.hp*10;
                        score+=pts;
                        store.addScore(pts);
                        explosions.push(new Explosion(e.x,e.y,e.col));
                    }
                }
            }
        }

        // ── 6. Enemy bullets ──────────────────────────────────────────────
        for(const b of eBullets){
            b.update(); b.draw();
            if(!b.dead && player.inv===0 && hit(b.x,b.y,6,8,player.x,player.y,player.w,player.h)){
                b.dead=true;
                explosions.push(new Explosion(player.x,player.y,'#F44336'));
                lives--; player.inv=90;
            }
        }

        // ── 7. Explosions ─────────────────────────────────────────────────
        for(const ex of explosions){ ex.update(); ex.draw(); }

        // ── 8. HUD & win/lose ─────────────────────────────────────────────
        drawHUD();
        if(lives<=0){ store.save(); gameState='gameover'; }
        // Periodic auto-save every 300 frames (~5s at 60fps) during gameplay
        if(frame%300===0) store.save();

    } catch(err){
        console.error('[GameLoop Error]', err);
    }
}

// ── Input ─────────────────────────────────────────────────────────────────────
document.addEventListener('keydown',e=>{
    keys[e.key]=true;
    if(e.key===' '){
        e.preventDefault();
        if(gameState==='menu') resetGame();
        else if(gameState==='gameover'){ store.save(); gameState='menu'; }
    }
    if(e.key==='s'||e.key==='S'){
        if(gameState==='menu'||gameState==='gameover'){ store.save(); gameState='shop'; }
        else if(gameState==='shop') gameState='menu';
    }
    if(e.key==='Escape'&&gameState==='shop') gameState='menu';
});
document.addEventListener('keyup',e=>{ keys[e.key]=false; });

canvas.addEventListener('click',e=>{
    const r=canvas.getBoundingClientRect();
    const mx=(e.clientX-r.left)*(W/r.width);
    const my=(e.clientY-r.top)*(H/r.height);
    if(gameState==='menu'){
        ['easy','medium','hard'].forEach((d,i)=>{
            const bx=W/2-130+i*130,by=196,bh=36;
            if(mx>=bx-50&&mx<=bx+50&&my>=by&&my<=by+bh) diffMode=d;
        });
        if(mx>=W/2-100&&mx<=W/2+100&&my>=355&&my<=395){ store.save(); gameState='shop'; }

        // Menu Save/Load handlers
        const bx1 = W/2 - 130, bx2 = W/2 + 20, by = 410, bw = 110, bh = 30;
        if(mx>=bx1 && mx<=bx1+bw && my>=by && my<=by+bh) {
            // Export: Just trigger the download directly (simplest for user)
            store.downloadSave();
        }
        if(mx>=bx2 && mx<=bx2+bw && my>=by && my<=by+bh) {
            // Import: Trigger file upload
            const input = document.createElement('input');
            input.type = 'file';
            input.accept = '.pshooter';
            input.onchange = e => {
                const file = e.target.files[0];
                if (!file) return;
                const reader = new FileReader();
                reader.onload = re => {
                    const res = store.importFile(re.target.result);
                    alert(res.ok ? '✅ Save imported successfully!' : '❌ Error: ' + res.reason);
                };
                reader.readAsArrayBuffer(file);
            };
            input.click();
        }
    }
    if(gameState==='shop'){
        const res=handleShopClick(mx,my);
        if(res==='close') gameState='menu';
    }
});

// ── Boot ──────────────────────────────────────────────────────────────────────
window.addEventListener('load',()=>{
    initStars();
    enemies=[]; pBullets=[]; eBullets=[]; explosions=[];
    score=0; lives=3; frame=0; spawnTimer=0;
    gameLoop();
});

// Save on ANY exit: refresh, back button, tab close
// This guarantees points/score are never lost mid-session
window.addEventListener('beforeunload',()=>{ store.save(); });
