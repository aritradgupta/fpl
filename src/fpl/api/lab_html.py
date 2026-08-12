"""Small self-contained browser UI for the solver comparison lab."""

SOLVER_LAB_HTML = """
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>FPL Solver Lab</title>
<style>
:root{color-scheme:dark;font-family:Inter,system-ui,sans-serif}body{margin:0;background:#101522;color:#edf2f7}
main{max-width:1180px;margin:auto;padding:32px 20px 60px}.muted{color:#9aa8bc}
.panel{background:#182235;border:1px solid #2b3a52;border-radius:12px;padding:18px;margin:20px 0}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px}
label{display:grid;gap:6px;color:#b9c7d9;font-size:13px}input,select{box-sizing:border-box;border:1px solid #3a4d6a;border-radius:7px;padding:9px;background:#101827;color:white}
.player-picks{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin:16px 0}.player-picks select{min-height:180px}
.solvers{display:flex;flex-wrap:wrap;gap:16px;margin:14px 0}.solvers label{display:flex;align-items:center;gap:7px}.solvers input{width:auto}
button{cursor:pointer;border:0;border-radius:8px;padding:11px 18px;background:#35c48a;color:#07130d;font-weight:700}button:disabled{opacity:.6;cursor:wait}
table{width:100%;border-collapse:collapse;margin-top:14px}th,td{padding:11px 9px;border-bottom:1px solid #2b3a52;text-align:left}th{color:#9fb4d1;font-size:12px;text-transform:uppercase}.ok{color:#55e6a4}.failed{color:#ff8f8f}pre{white-space:pre-wrap;color:#ffadad}
</style>
</head>
<body><main>
<h1>FPL Solver Lab</h1>
<div class="muted">Run the same player pool and official fixtures through every solver and compare outcomes.</div>
<section class="panel">
<div class="solvers">
<label><input type="checkbox" name="solver" value="single_period" checked> Single-period ILP</label>
<label><input type="checkbox" name="solver" value="multi_period" checked> Multi-period ILP</label>
<label><input type="checkbox" name="solver" value="stochastic" checked> Stochastic</label>
<label><input type="checkbox" name="solver" value="genetic" checked> Genetic</label>
</div>
<div class="grid">
<label>Budget (£m)<input id="budget" type="number" value="100" min="50" max="120" step=".1"></label>
<label>Club limit<input id="club_limit" type="number" value="3" min="1" max="5"></label>
<label>Gameweek<input id="gameweek" type="number" value="1" min="1" max="50"></label>
<label>Horizon weeks<input id="horizon_weeks" type="number" value="3" min="1" max="10"></label>
<label>Risk aversion<input id="risk_aversion" type="number" value=".15" min="0" max="2" step=".05"></label>
<label>Scenarios<input id="num_scenarios" type="number" value="500" min="10" max="10000"></label>
<label>GA generations<input id="generations" type="number" value="50" min="1" max="500"></label>
<label>GA population<input id="population_size" type="number" value="60" min="10" max="1000"></label>
<label>Chip<select id="chip"><option value="none">None</option><option value="bboost">Bench Boost</option><option value="3xc">Triple Captain</option></select></label>
</div>
<div class="player-picks">
<label>Include / lock players (Ctrl/Cmd-click for multiple)<select id="lock_players" multiple></select></label>
<label>Exclude players (Ctrl/Cmd-click for multiple)<select id="exclude_players" multiple></select></label>
</div>
<p><label><input id="use_gpu" type="checkbox" checked> Use GPU for stochastic scenarios when available</label></p>
<button id="run">Run selected solvers</button> <span id="message" class="muted"></span>
</section>
<section class="panel"><div id="meta" class="muted">No runs yet.</div><div id="results"></div></section>
</main>
<script>
const $=id=>document.getElementById(id), n=id=>Number($(id).value);
function selected_ids(id){return [...$(id).selectedOptions].map(x=>Number(x.value))}
async function load_players(){
 const response=await fetch("/api/lab/players");const players=await response.json();
 ["lock_players","exclude_players"].forEach(id=>{
  $(id).innerHTML=players.map(p=>"<option value='"+p.id+"'>"+p.web_name+" · "+p.position+" · "+p.team+" · £"+Number(p.cost).toFixed(1)+"m</option>").join("");
 });
}
load_players().catch(error=>{$("message").textContent="Unable to load player options: "+error.message});
$("run").onclick=async()=>{
 const solvers=[...document.querySelectorAll('input[name="solver"]:checked')].map(x=>x.value);
 if(!solvers.length){$("message").textContent="Select at least one solver.";return}
 const payload={solver_types:solvers,budget:n("budget"),club_limit:n("club_limit"),gameweek:n("gameweek"),horizon_weeks:n("horizon_weeks"),chip:$("chip").value,risk_aversion:n("risk_aversion"),num_scenarios:n("num_scenarios"),use_gpu:$("use_gpu").checked,generations:n("generations"),population_size:n("population_size"),seed:42,lock_player_ids:selected_ids("lock_players"),exclude_player_ids:selected_ids("exclude_players")};
 $("run").disabled=true;$("message").textContent="Running...";
 try{
  const response=await fetch("/api/lab/run",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(payload)});
  const data=await response.json();if(!response.ok)throw new Error(data.detail||"Lab request failed");
  $("meta").textContent=data.player_count+" players · "+data.fixture_rows+" fixture rows";
  let rows=data.results.map(r=>{
   const players=r.players.map(p=>"<li><strong>"+p.name+"</strong> · "+p.position+" · "+p.team+" · £"+p.cost.toFixed(1)+"m · "+p.role+" <small>(ID "+p.player_id+")</small></li>").join("");
   const squad="<details><summary>Show "+r.players.length+" players</summary><ol>"+players+"</ol></details>";
   return "<tr><td>"+r.solver+"</td><td class='"+(r.status==="ok"?"ok":"failed")+"'>"+r.status+"</td><td>"+r.runtime_seconds.toFixed(2)+"s</td><td>"+(r.total_expected_points??"—")+"</td><td>"+(r.total_cost??"—")+"</td><td>"+(r.captain??"—")+"</td><td>"+squad+"</td></tr>"+(r.error?"<tr><td colspan='7'><pre>"+r.error+"</pre></td></tr>":"");
  }).join("");
  $("results").innerHTML="<table><thead><tr><th>Solver</th><th>Status</th><th>Runtime</th><th>XI xP</th><th>Cost</th><th>Captain</th><th>Players</th></tr></thead><tbody>"+rows+"</tbody></table>";
  $("message").textContent="Complete";
 }catch(error){$("message").textContent=error.message}finally{$("run").disabled=false}
};
</script></body></html>
"""
