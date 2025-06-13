import { Component, OnInit } from '@angular/core';

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.css']
})
export class AppComponent implements OnInit {
  activeTab = 'record';
  sidebarCollapsed = false;
  asrEngine = 'Vosk';
  consentRecorded = false;
  patientName = '';
  consentInfo = '';
  isRecording = false;
  recordingTime = '00:00';
  finalText = '';
  partialText = '';
  private recordTimer?: any;
  private recordStart = 0;

  ngOnInit(): void {}

  switchTab(name: string) {
    this.activeTab = name;
  }

  toggleSidebar() {
    this.sidebarCollapsed = !this.sidebarCollapsed;
  }

  onAsrChange(value: string) {
    this.asrEngine = value;
  }

  documentConsent() {
    this.consentRecorded = true;
    const now = new Date();
    this.consentInfo = `${this.patientName} - ${now.toLocaleDateString()} ${now.toLocaleTimeString()}`;
  }

  startRecording() {
    this.isRecording = true;
    this.recordStart = Date.now();
    this.updateTime();
    this.recordTimer = setInterval(() => this.updateTime(), 1000);
    this.simulateTranscription();
  }

  private updateTime() {
    const elapsed = Math.floor((Date.now() - this.recordStart)/1000);
    const m = Math.floor(elapsed/60).toString().padStart(2,'0');
    const s = (elapsed%60).toString().padStart(2,'0');
    this.recordingTime = `${m}:${s}`;
  }

  private simulateTranscription() {
    const phrases = [
      'The patient presents with',
      'chief complaint of',
      'persistent headache',
      'for the past three days.',
      'No fever or nausea reported.',
      'Blood pressure is',
      '120 over 80.',
      'Heart rate is normal.'
    ];
    let index = 0;
    const interval = setInterval(() => {
      if (!this.isRecording || index >= phrases.length) {
        clearInterval(interval);
        return;
      }
      this.finalText += (this.finalText ? ' ' : '') + phrases[index];
      if (index < phrases.length - 1) {
        this.partialText = '...';
      } else {
        this.partialText = '';
      }
      index++;
    }, 2000);
  }
}
