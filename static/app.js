const analyzeBtn = document.getElementById('analyzeBtn');
const statusBox = document.getElementById('status');
const result = document.getElementById('result');

function showStatus(message, kind = 'error') {
  statusBox.hidden = false;
  statusBox.className = `status ${kind}`;
  statusBox.textContent = message;
}

function clearStatus() {
  statusBox.hidden = true;
  statusBox.textContent = '';
  statusBox.className = 'status';
}

function setTags(elementId, items, emptyText) {
  const el = document.getElementById(elementId);
  el.innerHTML = '';
  if (!items || items.length === 0) {
    el.innerHTML = `<span class="tag empty">${emptyText}</span>`;
    return;
  }
  items.forEach(item => {
    const span = document.createElement('span');
    span.className = 'tag';
    span.textContent = item;
    el.appendChild(span);
  });
}

function renderPredictions(items, recommendedRole) {
  const el = document.getElementById('predictions');
  el.innerHTML = '';
  items.forEach((item, index) => {
    const row = document.createElement('div');
    row.className = `rank ${item.role === recommendedRole ? 'winner' : ''}`;
    row.innerHTML = `
      <div class="rank-num">${index + 1}</div>
      <div class="rank-name">
        <strong>${escapeHtml(item.role)}</strong>
        <span>ML ${item.mlProbability.toFixed(1)}% • Skills ${item.skillCoverage.toFixed(1)}%</span>
      </div>
      <div class="bar"><div class="fill" style="width:${Math.min(item.score, 100)}%"></div></div>
      <div class="rank-score">${item.score.toFixed(1)}%</div>`;
    el.appendChild(row);
  });
}

function renderList(elementId, items) {
  const el = document.getElementById(elementId);
  el.innerHTML = '';
  items.forEach(item => {
    const li = document.createElement('li');
    li.textContent = item;
    el.appendChild(li);
  });
}

function escapeHtml(text) {
  return String(text).replace(/[&<>'"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[c]));
}

function renderReasons(reasons) {
  const el = document.getElementById('reasons');
  el.innerHTML = '';
  reasons.forEach(reason => {
    const p = document.createElement('p');
    p.textContent = reason;
    el.appendChild(p);
  });
}

analyzeBtn.addEventListener('click', async () => {
  clearStatus();
  analyzeBtn.disabled = true;
  analyzeBtn.textContent = 'Analyzing profile…';

  try {
    const form = new FormData();
    ['name', 'education', 'skills', 'interests', 'experience', 'targetRole'].forEach(id => {
      form.append(id, document.getElementById(id).value.trim());
    });
    const resume = document.getElementById('resume').files[0];
    if (resume) form.append('resume', resume);

    const response = await fetch('/api/analyze', { method: 'POST', body: form });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || 'Unable to analyze profile.');

    document.getElementById('headline').textContent = `${data.recommendedRole} is your strongest match`;
    document.getElementById('subheadline').textContent = `${data.profile.name} • ${data.profile.education}`;
    document.getElementById('profileBadge').textContent = `Profile completeness ${data.profile.completeness}%`;
    document.getElementById('hybridScore').textContent = `${data.hybridScore.toFixed(1)}%`;
    document.getElementById('confidence').textContent = `${data.modelProbability.toFixed(1)}%`;
    document.getElementById('skillScore').textContent = `${data.skillMatchScore.toFixed(1)}%`;
    document.getElementById('selectedRole').textContent = data.recommendedRole;
    document.getElementById('mlRole').textContent = data.mlRecommendedRole;
    document.getElementById('completeness').textContent = `${data.profile.completeness}%`;
    document.getElementById('skillCount').textContent = data.detectedSkills.length;
    document.getElementById('resumeCount').textContent = data.resumeSkills.length;
    document.getElementById('modelNote').textContent = `${data.note} Training data note: ${data.model.datasetNote}`;
    document.getElementById('metricAccuracy').textContent = `${data.model.holdoutAccuracy}%`;
    document.getElementById('metricF1').textContent = `${data.model.holdoutF1}%`;
    document.getElementById('metricRows').textContent = data.model.trainingRows;

    renderReasons(data.reasons);
    renderPredictions(data.predictions, data.recommendedRole);
    setTags('matchedSkills', data.matchedSkills, 'No role-specific skills matched yet.');
    setTags('skillGaps', data.skillGaps, 'No priority gaps detected.');
    setTags('resumeSkills', data.resumeSkills, 'No resume uploaded or no supported skills were extracted.');
    renderList('roadmap', data.roadmap);
    renderList('projects', data.projects);

    result.hidden = false;
    result.scrollIntoView({ behavior: 'smooth', block: 'start' });
  } catch (error) {
    result.hidden = true;
    showStatus(error.message || 'Something went wrong.');
  } finally {
    analyzeBtn.disabled = false;
    analyzeBtn.textContent = 'Analyze My Career →';
  }
});
