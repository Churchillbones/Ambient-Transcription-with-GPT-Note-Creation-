import { Component, OnInit, OnDestroy, ChangeDetectorRef } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { interval, Subscription } from 'rxjs';
import { TranscriptionService, NoteRequest } from './transcription.service';

interface Config {
  azureApiKey: string;
  azureEndpoint: string;
  noteModel: string;
  localModel: string;
  asrEngine: string;
  voskModel: string;
  whisperModel: string;
  encryptRecordings: boolean;
  selectedTemplate: string;
}

interface ComparisonModel {
  type: string;
  size: string;
}

interface Comparison {
  model1: ComparisonModel;
  model2: ComparisonModel;
}

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.css']
})
export class AppComponent implements OnInit, OnDestroy {
  // UI State
  activeTab = 'record';
  sidebarCollapsed = false;
  currentStep = 0;
  showResults = false;

  // Template options
  templateOptions = [
    { value: 'SOAP', label: 'SOAP Note', description: 'Standard Subjective, Objective, Assessment, Plan' },
    { value: 'PrimaryCare', label: 'Primary Care Visit', description: 'Comprehensive primary care documentation' },
    { value: 'Psychiatric Assessment', label: 'Psychiatric Assessment', description: 'Mental health evaluation with MSE and risk assessment' },
    { value: 'Discharge Summary', label: 'Discharge Summary', description: 'Hospital discharge documentation' },
    { value: 'Operative Note', label: 'Operative Note', description: 'Surgical procedure documentation' },
    { value: 'Biopsychosocial', label: 'Biopsychosocial Assessment', description: 'Mental health biopsychosocial format' },
    { value: 'Consultation Note', label: 'Consultation Note', description: 'Specialist consultation documentation' },
    { value: 'Well-Child Visit', label: 'Well-Child Visit', description: 'Pediatric wellness examination' },
    { value: 'Emergency Department', label: 'Emergency Department', description: 'ED visit documentation' },
    { value: 'Progress Note', label: 'Progress Note', description: 'Follow-up visit documentation' }
  ];

  // Local model options
  localModelOptions = [
    { value: 'gemma3-4b', label: 'Gemma 3 4B', description: 'Fast, efficient model good for medical tasks (~4GB)' },
    { value: 'deepseek-r1-14b', label: 'DeepSeek R1 14B', description: 'Large, powerful model for complex medical documentation (~14GB)' },
    { value: 'llama3', label: 'Llama 3', description: 'Meta\'s general purpose model' },
    { value: 'phi4', label: 'Phi 4', description: 'Microsoft\'s compact model' }
  ];

  // Configuration
  config: Config = {
    azureApiKey: '',
    azureEndpoint: 'https://your-resource.openai.azure.com/',
    noteModel: 'azure',
    localModel: 'gemma3-4b',
    asrEngine: 'vosk',
    voskModel: 'vosk-model-en-us-0.22',
    whisperModel: 'tiny',
    encryptRecordings: true,
    selectedTemplate: 'SOAP'
  };

  // Consent
  consentDocumented = false;
  patientName = '';
  consentDate = new Date();

  // Recording
  recordingMode = 'traditional';
  isRecording = false;
  recordingTime = '00:00';
  recordingStartTime = 0;
  recordingInterval?: Subscription;
  mediaRecorder?: MediaRecorder;
  audioChunks: Blob[] = [];

  // Realtime streaming helpers
  realtimeSocket?: WebSocket;
  audioContext?: AudioContext;
  scriptNode?: ScriptProcessorNode;

  // Transcription
  finalTranscriptionText = '';
  partialTranscriptionText = '';
  lowConfidenceWords: string[] = [];

  // Results
  rawTranscription = '';
  cleanedTranscription = '';
  speakerDiarization = '';
  generatedNote = '';

  // File Upload
  selectedFile?: File;
  selectedFileUrl: string | null = null;

  // Model Comparison
  comparison: Comparison = {
    model1: { type: 'vosk', size: 'small' },
    model2: { type: 'whisper', size: 'tiny' }
  };

  constructor(
    private http: HttpClient,
    private transcriptionService: TranscriptionService,
    private cdr: ChangeDetectorRef
  ) {}

  ngOnInit() {
    this.loadConfiguration();
  }

  ngOnDestroy() {
    if (this.recordingInterval) {
      this.recordingInterval.unsubscribe();
    }
    if (this.isRecording) {
      this.stopRecording();
    }
    if (this.selectedFileUrl) {
      URL.revokeObjectURL(this.selectedFileUrl);
    }
  }

  // Configuration Management
  loadConfiguration() {
    const savedConfig = localStorage.getItem('medicalTranscriptionConfig');
    if (savedConfig) {
      const parsed = JSON.parse(savedConfig);
      // Force correct model names, ignore old cached values
      if (parsed.voskModel === 'vosk_small' || parsed.voskModel === 'vosk_large') {
        parsed.voskModel = 'vosk-model-en-us-0.22';
      }
      this.config = { ...this.config, ...parsed };
    }
    // Always save the corrected config
    this.saveConfiguration();
  }

  saveConfiguration() {
    localStorage.setItem('medicalTranscriptionConfig', JSON.stringify(this.config));
  }

  resetConfiguration() {
    // Clear localStorage and reset to defaults
    localStorage.removeItem('medicalTranscriptionConfig');
    this.config = {
      azureApiKey: '',
      azureEndpoint: 'https://your-resource.openai.azure.com/',
      noteModel: 'azure',
      localModel: 'gemma3-4b',
      asrEngine: 'vosk',
      voskModel: 'vosk-model-en-us-0.22',
      whisperModel: 'tiny',
      encryptRecordings: true,
      selectedTemplate: 'SOAP'
    };
    this.saveConfiguration();
    alert('Configuration reset to defaults');
  }

  onAsrEngineChange() {
    this.saveConfiguration();
  }

  getTemplateDescription(): string {
    const selectedTemplate = this.templateOptions.find(t => t.value === this.config.selectedTemplate);
    return selectedTemplate ? selectedTemplate.description : '';
  }

  getLocalModelDescription(): string {
    const selectedModel = this.localModelOptions.find(m => m.value === this.config.localModel);
    return selectedModel ? selectedModel.description : '';
  }

  // UI Navigation
  switchTab(tabName: string) {
    this.activeTab = tabName;
  }

  toggleSidebar() {
    this.sidebarCollapsed = !this.sidebarCollapsed;
  }

  // Consent Management
  documentConsent() {
    if (!this.patientName.trim()) {
      alert('Please enter patient name');
      return;
    }
    this.consentDocumented = true;
    this.consentDate = new Date();
  }

  // Recording Functions
  async startRecording() {
    try {
      // Clear previous transcription text and results
      this.finalTranscriptionText = '';
      this.partialTranscriptionText = '';
      this.lowConfidenceWords = [];
      this.rawTranscription = '';
      this.cleanedTranscription = '';
      this.generatedNote = '';
      this.showResults = false;

      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      this.mediaRecorder = new MediaRecorder(stream);
      this.audioChunks = [];

      this.mediaRecorder.ondataavailable = (event) => {
        this.audioChunks.push(event.data);
      };

      this.mediaRecorder.onstop = () => {
        // In 'traditional' mode, process the full audio blob.
        // In 'realtime' mode, the transcription is already finalized, so just trigger post-processing.
        if (this.recordingMode === 'traditional') {
          const audioBlob = new Blob(this.audioChunks, { type: 'audio/wav' });
          this.processRecording(audioBlob);
        } else {
          this.rawTranscription = this.finalTranscriptionText.trim();
          if (this.rawTranscription) {
            this.updateProgress(2);
            this.showResults = true; // Show results immediately after transcription
            this.switchTab('record'); // Ensure user is on the record tab to see results
            this.cleanTranscription().then(() => {
              this.generateSpeakerDiarization();
              this.updateProgress(3);
              this.generateMedicalNote().then(() => {
                this.updateProgress(4);
              }).catch((noteError) => {
                console.error('Error generating note in realtime mode:', noteError);
                this.generatedNote = 'Error generating note. Please check your Azure API configuration.';
                this.updateProgress(4);
              });
            }).catch((cleanError) => {
              console.error('Error cleaning transcription in realtime mode:', cleanError);
              this.cleanedTranscription = this.rawTranscription;
              this.generateSpeakerDiarization();
              this.updateProgress(3);
            });
          } else {
            this.rawTranscription = 'No transcription available from real-time session.';
            this.showResults = true;
            this.switchTab('record'); // Ensure user sees the message
          }
        }
      };

      this.mediaRecorder.start();
      this.isRecording = true;
      this.recordingStartTime = Date.now();
      this.startRecordingTimer();

      if (this.recordingMode === 'realtime') {
        // Pass the already-acquired stream to avoid a second permission prompt
        this.startRealtimeTranscription(stream);
      }

      this.updateProgress(0);
    } catch (error) {
      console.error('Error starting recording:', error);
      alert('Error accessing microphone. Please check permissions.');
    }
  }

  pauseRecording() {
    if (this.mediaRecorder && this.isRecording) {
      this.mediaRecorder.pause();
    }
  }

  stopRecording() {
    if (this.mediaRecorder && this.isRecording) {
      this.mediaRecorder.stop();
      this.mediaRecorder.stream.getTracks().forEach(track => track.stop());
      this.isRecording = false;
      this.stopRecordingTimer();
      
      // Handle real-time transcription completion
      if (this.recordingMode === 'realtime' && this.finalTranscriptionText) {
        // Transfer real-time transcription to main transcription and process it
        this.rawTranscription = this.finalTranscriptionText;
        this.showResults = true;
        this.switchTab('record'); // Switch back to main view
        this.showTranscriptionComplete();
        
        // Start with step 1 (Transcribe) since recording is done
        this.updateProgress(1);
        
        // Process the real-time transcription through the full pipeline
        this.processRealtimeTranscription();
      } else {
        // For traditional recording, set to step 1 (Transcribe)
        this.updateProgress(1);
      }
      
      // Cleanup realtime stream
      if (this.realtimeSocket) {
        this.realtimeSocket.close();
        this.realtimeSocket = undefined;
      }
      if (this.scriptNode) {
        this.scriptNode.disconnect();
        this.scriptNode = undefined as any;
      }
      if (this.audioContext) {
        this.audioContext.close();
        this.audioContext = undefined;
      }
    }
  }

  private startRecordingTimer() {
    this.recordingInterval = interval(1000).subscribe(() => {
      const elapsed = Math.floor((Date.now() - this.recordingStartTime) / 1000);
      const minutes = Math.floor(elapsed / 60).toString().padStart(2, '0');
      const seconds = (elapsed % 60).toString().padStart(2, '0');
      this.recordingTime = `${minutes}:${seconds}`;
    });
  }

  private stopRecordingTimer() {
    if (this.recordingInterval) {
      this.recordingInterval.unsubscribe();
    }
  }

  private startRealtimeTranscription(stream: MediaStream) {
    const SAMPLE_RATE = 16000;

    // Use the actual model folder name for WebSocket
    const modelParam = this.config.voskModel || 'vosk-model-en-us-0.22';
    const wsUrl = `ws://${location.hostname}:8000/ws/vosk?model=${encodeURIComponent(modelParam)}`;
    console.log('Connecting to WebSocket:', wsUrl);
    this.realtimeSocket = new WebSocket(wsUrl);

    this.realtimeSocket.onopen = () => {
      console.log('WebSocket connected successfully');
    };

    this.realtimeSocket.onerror = (error) => {
      console.error('WebSocket error:', error);
      alert('WebSocket connection failed. Ensure the backend is running and accessible.');
    };

    this.realtimeSocket.onclose = (event) => {
      console.log('WebSocket closed:', event.code, event.reason);
      // If closed unexpectedly, clean up audio resources to prevent errors
      if (this.scriptNode) {
        this.scriptNode.disconnect();
      }
      if (this.audioContext && this.audioContext.state !== 'closed') {
        this.audioContext.close();
      }
    };

    this.realtimeSocket.onmessage = ({ data }) => {
      const msg = JSON.parse(data);
      console.log('WebSocket message received:', msg);
      
      if (msg.type === 'partial') {
        this.partialTranscriptionText = msg.text;
      }
      if (msg.type === 'final') {
        if (msg.text) {
          this.finalTranscriptionText += (this.finalTranscriptionText ? ' ' : '') + msg.text;
          console.log('Final transcription updated:', this.finalTranscriptionText);
          
          // Check for low confidence words if result contains word details
          if (msg.result && msg.result.length > 0) {
            const lowConfWords = msg.result
              .filter((word: any) => word.conf && word.conf < 0.8)
              .map((word: any) => word.word);
            if (lowConfWords.length > 0) {
              this.lowConfidenceWords = [...new Set([...this.lowConfidenceWords, ...lowConfWords])];
            }
          }
        }
        this.partialTranscriptionText = '';
      }
    };

    // Use the stream passed from startRecording() to avoid a second permission prompt
    try {
      this.audioContext = new ((window as any).AudioContext || (window as any).webkitAudioContext)({ sampleRate: SAMPLE_RATE });
      const source = this.audioContext.createMediaStreamSource(stream);
      // Deprecated but widely supported – adequate for demo purposes
      this.scriptNode = this.audioContext.createScriptProcessor(4096, 1, 1);

      this.scriptNode.onaudioprocess = (event) => {
        // Don't process audio if recording has stopped or WebSocket is not open
        if (!this.isRecording || !this.realtimeSocket || this.realtimeSocket.readyState !== WebSocket.OPEN) {
          return;
        }

        const input = event.inputBuffer.getChannelData(0);
        const buffer = new ArrayBuffer(input.length * 2);
        const view = new DataView(buffer);
        for (let i = 0; i < input.length; i++) {
          const s = Math.max(-1, Math.min(1, input[i]));
          view.setInt16(i * 2, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
        }
        this.realtimeSocket.send(buffer);
      };

      source.connect(this.scriptNode);
      // We connect the script node to the destination to avoid audio being silenced in some browsers.
      this.scriptNode.connect(this.audioContext.destination);
    } catch (err) {
      console.error('Error setting up AudioContext for real-time transcription:', err);
      alert('Failed to set up real-time audio processing.');
    }
  }

  private async processRecording(audioBlob: Blob) {
    const audioFile = new File([audioBlob], 'recording.' + (audioBlob.type.split('/')[1] || 'webm'), { type: audioBlob.type });

    try {
      this.updateProgress(1);
      const response = await this.transcriptionService.transcribeAudio(
        audioFile, 
        this.getSelectedModel(), 
        'en-US',
        this.getVoskModelPath()
      ).toPromise();
      
      if (response) {
        this.rawTranscription = response.transcript;
        this.updateProgress(2);
        this.showResults = true; // Show results immediately after transcription
        this.switchTab('record'); // Ensure user is on the record tab to see results
        this.showTranscriptionComplete();
        
        // Clean transcription
        try {
          await this.cleanTranscription();
          this.updateProgress(3);
        } catch (cleanError) {
          console.error('Error cleaning transcription:', cleanError);
          this.cleanedTranscription = this.rawTranscription; // Fallback to raw transcription
          this.updateProgress(3);
        }

        // Generate speaker diarization (simulate for now)
        this.generateSpeakerDiarization();
        
        // Generate note
        try {
          await this.generateMedicalNote();
          this.updateProgress(4);
        } catch (noteError) {
          console.error('Error generating note:', noteError);
          this.generatedNote = 'Error generating note. Please check your Azure API configuration.';
          this.updateProgress(4);
        }
      } else {
        // Even if no response, show some feedback
        this.rawTranscription = 'No transcription result received.';
        this.showResults = true;
        this.switchTab('record'); // Ensure user sees the error message
      }
    } catch (error) {
      console.error('Error processing recording:', error);
      // Show results even on error so user can see what happened
      this.rawTranscription = `Error: ${error?.error?.error || error.message || 'Unknown transcription error'}`;
      this.showResults = true;
      this.updateProgress(2);
      this.switchTab('record'); // Ensure user sees the error message
    }
  }

  private getSelectedModel(): string {
    if (this.config.asrEngine === 'vosk') {
      return 'vosk';
    } else if (this.config.asrEngine === 'whisper') {
      return `whisper_${this.config.whisperModel}`;
    }
    return 'vosk';
  }

  private getVoskModelPath(): string | undefined {
    if (this.config.asrEngine === 'vosk') {
      return `app_data/models/${this.config.voskModel}`;
    }
    return undefined;
  }

  private async cleanTranscription() {
    // Simulate transcription cleaning
    this.cleanedTranscription = this.rawTranscription
      .replace(/um/g, '')
      .replace(/uh/g, '')
      .replace(/\s+/g, ' ')
      .trim();
  }

  private generateSpeakerDiarization() {
    // Generate a simple speaker diarization based on the transcription
    if (this.rawTranscription && this.rawTranscription.trim()) {
      // Simple simulation - split by sentences and alternate speakers
      const sentences = this.rawTranscription.split(/[.!?]+/).filter(s => s.trim());
      let diarization = '';
      
      sentences.forEach((sentence, index) => {
        if (sentence.trim()) {
          const speaker = index % 2 === 0 ? 'Doctor' : 'Patient';
          diarization += `<p><strong>${speaker}:</strong> ${sentence.trim()}.</p>`;
        }
      });
      
      this.speakerDiarization = diarization || '<p><strong>Doctor:</strong> How are you feeling today?</p><p><strong>Patient:</strong> I\'ve been experiencing some discomfort...</p>';
    } else {
      this.speakerDiarization = '<p><strong>Doctor:</strong> How are you feeling today?</p><p><strong>Patient:</strong> I\'ve been experiencing some discomfort...</p>';
    }
  }

  private async generateMedicalNote() {
    try {
      const noteRequest: NoteRequest = {
        transcript: this.rawTranscription,
        template: this.config.selectedTemplate,
        api_key: this.config.azureApiKey,
        endpoint: this.config.azureEndpoint,
        model: 'gpt-4o',
        use_local: this.config.noteModel === 'local',
        local_model: this.config.localModel
      };

      const response = await this.transcriptionService.generateNote(noteRequest).toPromise();
      
      if (response) {
        this.generatedNote = response.note;
      }
    } catch (error) {
      console.error('Error generating note:', error);
      this.generatedNote = '<h4>SOAP Note</h4><p><strong>Subjective:</strong> Patient reports...</p><p><strong>Objective:</strong> Vital signs...</p><p><strong>Assessment:</strong> Clinical impression...</p><p><strong>Plan:</strong> Treatment recommendations...</p>';
    }
  }

  // File Upload Functions
  onFileSelected(event: any) {
    const file: File = event.target.files[0];
    if (file) {
      this.selectedFile = file;
      if (this.selectedFileUrl) {
        URL.revokeObjectURL(this.selectedFileUrl);
      }
      this.selectedFileUrl = URL.createObjectURL(file);
    }
  }

  onDragOver(event: DragEvent) {
    event.preventDefault();
    const uploadArea = event.target as HTMLElement;
    uploadArea.style.borderColor = 'var(--primary-color)';
    uploadArea.style.background = 'rgba(37, 99, 235, 0.05)';
  }

  onDragLeave(event: DragEvent) {
    const uploadArea = event.target as HTMLElement;
    uploadArea.style.borderColor = 'var(--border-color)';
    uploadArea.style.background = 'transparent';
  }

  onDrop(event: DragEvent) {
    event.preventDefault();
    const uploadArea = event.target as HTMLElement;
    uploadArea.style.borderColor = 'var(--border-color)';
    uploadArea.style.background = 'transparent';

    const files = event.dataTransfer?.files;
    if (files && files.length > 0) {
      this.selectedFile = files[0];
      if (this.selectedFileUrl) {
        URL.revokeObjectURL(this.selectedFileUrl);
      }
      this.selectedFileUrl = URL.createObjectURL(files[0]);
    }
  }

  async uploadAndTranscribe() {
    if (!this.selectedFile) return;

    try {
      this.updateProgress(1);
      const response = await this.transcriptionService.transcribeAudio(
        this.selectedFile,
        this.getSelectedModel(),
        'en-US',
        this.getVoskModelPath()
      ).toPromise();
      
      if (response) {
        this.rawTranscription = response.transcript;
        this.updateProgress(2);
        
        await this.cleanTranscription();
        this.updateProgress(3);
        
        await this.generateMedicalNote();
        this.updateProgress(4);
        
        this.showResults = true;
      }
    } catch (error) {
      console.error('Error uploading file:', error);
      alert(`Error uploading file: ${error?.error?.error || error.message || 'Unknown error'}`);
    }
  }

  // Utility Functions
  updateProgress(step: number) {
    this.currentStep = step;
    // Manually trigger change detection to ensure UI updates immediately
    this.cdr.detectChanges();
  }

  copyToClipboard(text: string) {
    if (text) {
      navigator.clipboard.writeText(text.replace(/<[^>]*>/g, '')).then(() => {
        // Could show a toast notification here
        console.log('Copied to clipboard');
      });
    }
  }

  downloadText(text: string, filename: string) {
    if (text) {
      const cleanText = text.replace(/<[^>]*>/g, '');
      const blob = new Blob([cleanText], { type: 'text/plain' });
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = filename;
      a.click();
      window.URL.revokeObjectURL(url);
    }
  }

  editNote() {
    // Could open a modal or navigate to an edit view
    console.log('Edit note functionality');
  }

  exportResults() {
    const results = {
      patient: this.patientName,
      date: this.consentDate,
      rawTranscription: this.rawTranscription,
      cleanedTranscription: this.cleanedTranscription,
      generatedNote: this.generatedNote
    };
    
    const blob = new Blob([JSON.stringify(results, null, 2)], { type: 'application/json' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `medical-transcription-${new Date().toISOString().split('T')[0]}.json`;
    a.click();
    window.URL.revokeObjectURL(url);
  }

  runComparison() {
    console.log('Running model comparison:', this.comparison);
    // Implement model comparison logic
  }

  private async processRealtimeTranscription() {
    // Process real-time transcription through the full pipeline
    try {
      console.log('Starting real-time transcription processing pipeline...');
      
      // Step 2: Clean transcription (step 1 is transcribe, already done)
      try {
        console.log('Cleaning transcription...');
        await this.cleanTranscription();
        this.updateProgress(2); // Step 3: Clean
        await this.delay(200); // Small delay to ensure UI updates
      } catch (cleanError) {
        console.error('Error cleaning transcription:', cleanError);
        this.cleanedTranscription = this.rawTranscription; // Fallback to raw transcription
        this.updateProgress(2);
        await this.delay(200);
      }

      // Generate speaker diarization
      console.log('Generating speaker diarization...');
      this.generateSpeakerDiarization();
      
      // Step 4: Generate note
      try {
        console.log('Generating medical note...');
        await this.generateMedicalNote();
        this.updateProgress(3); // Step 4: Generate Note (0-indexed, so 3 = step 4)
        console.log('Real-time processing complete!');
      } catch (noteError) {
        console.error('Error generating note:', noteError);
        this.generatedNote = 'Error generating note. Please check your configuration.';
        this.updateProgress(3);
      }
      
      // Clear real-time transcription text since it's now processed
      this.finalTranscriptionText = '';
      this.partialTranscriptionText = '';
      
    } catch (error) {
      console.error('Error processing real-time transcription:', error);
      this.updateProgress(4);
    }
  }

  private delay(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  private showTranscriptionComplete() {
    // Add a brief visual indicator that transcription is complete
    console.log('✅ Transcription completed and results are now visible');
    // Could add a toast notification here in the future
  }
}
