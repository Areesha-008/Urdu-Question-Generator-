const $ = id => document.getElementById(id);
const examples = [['پاکستان کا دارالحکومت اسلام آباد ہے۔','اسلام آباد'],['علی لاہور میں رہتا ہے۔','علی'],['احمد نے تین کتابیں خریدیں۔','تین']];
document.querySelectorAll('[data-example]').forEach(button=>button.onclick=()=>{
  [$('sentence').value,$('answer').value]=examples[Number(button.dataset.example)];
});
fetch('/health').then(r=>r.json()).then(r=>$('health').textContent=r.ready?'Model ready':'Awaiting trained model').catch(()=>$('health').textContent='Connection unavailable');
$('form').onsubmit=async event=>{
  event.preventDefault();$('submit').disabled=true;$('message').textContent='Writing your questions…';
  try{
    const response=await fetch('/generate',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({sentence:$('sentence').value,answer:$('answer').value})});
    const data=await response.json();if(!response.ok)throw Error(typeof data.detail==='string'?data.detail:'Please check your sentence and answer.');
    $('greedy').textContent=data.greedy.text;$('beam').textContent=data.beam.text;
    $('message').textContent=data.greedy.ended&&data.beam.ended?'Questions ready.':'A question reached the length limit. Try a shorter sentence.';
  }catch(error){$('message').textContent=error.message;}finally{$('submit').disabled=false;}
};
