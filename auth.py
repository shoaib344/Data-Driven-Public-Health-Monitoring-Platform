"""
Authentication module for Streamlit Public Health Monitor
Handles user registration, login, and session management
"""
import streamlit as st
from typing import Optional, Dict, Any
from database_service import db_service, UserRole
from database_models import User, get_db

def init_session_state():
    """Initialize session state variables"""
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    if 'user' not in st.session_state:
        st.session_state.user = None
    if 'user_role' not in st.session_state:
        st.session_state.user_role = None
    if 'session_token' not in st.session_state:
        st.session_state.session_token = None

def show_login_page():
    """Display login page"""
    st.title("🏥 Public Health Monitor - Login")
    st.markdown("Sign in to access personalized health monitoring and alerts")
    
    # Create tabs for login and registration
    login_tab, register_tab = st.tabs(["Login", "Register"])
    
    with login_tab:
        st.subheader("Sign In")
        
        with st.form("login_form"):
            username = st.text_input("Username or Email")
            password = st.text_input("Password", type="password")
            submit_button = st.form_submit_button("Sign In")
            
            if submit_button:
                if username and password:
                    with st.spinner("Authenticating..."):
                        db = next(get_db())
                        user = db_service.auth.authenticate_user(db, username, password)
                        
                        if user:
                            # Create session (without problematic context.headers)
                            session = db_service.auth.create_session(
                                db, 
                                str(user.id),
                                ip_address='streamlit_client',
                                user_agent='streamlit_app'
                            )
                            
                            # Update session state
                            st.session_state.authenticated = True
                            st.session_state.user = {
                                'id': str(user.id),
                                'username': user.username,
                                'email': user.email,
                                'first_name': user.first_name,
                                'last_name': user.last_name,
                                'role': user.role.value,
                                'organization': user.organization
                            }
                            st.session_state.user_role = user.role
                            st.session_state.session_token = session.session_token
                            
                            st.success(f"Welcome back, {user.first_name or user.username}!")
                            st.rerun()
                        else:
                            st.error("Invalid username/email or password")
                        
                        db.close()
                else:
                    st.error("Please enter both username and password")
    
    with register_tab:
        st.subheader("Create Account")
        
        with st.form("register_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                reg_username = st.text_input("Username*")
                reg_email = st.text_input("Email*")
                reg_password = st.text_input("Password*", type="password")
                reg_first_name = st.text_input("First Name")
                
            with col2:
                reg_confirm_password = st.text_input("Confirm Password*", type="password")
                reg_last_name = st.text_input("Last Name")
                reg_organization = st.text_input("Organization")
                reg_phone = st.text_input("Phone Number")
            
            # Account type selection
            account_type = st.selectbox(
                "Account Type",
                options=["Public User", "Health Authority"],
                help="Public users can view dashboards and set personal alerts. Health authorities have access to detailed analytics."
            )
            
            # Terms acceptance
            agree_terms = st.checkbox("I agree to the Terms of Service and Privacy Policy*")
            
            register_button = st.form_submit_button("Create Account")
            
            if register_button:
                # Validation
                if not all([reg_username, reg_email, reg_password, reg_confirm_password]):
                    st.error("Please fill in all required fields marked with *")
                elif reg_password != reg_confirm_password:
                    st.error("Passwords do not match")
                elif len(reg_password) < 6:
                    st.error("Password must be at least 6 characters long")
                elif not agree_terms:
                    st.error("Please agree to the Terms of Service and Privacy Policy")
                else:
                    with st.spinner("Creating account..."):
                        db = next(get_db())
                        
                        # Determine user role
                        role = UserRole.HEALTH_AUTHORITY if account_type == "Health Authority" else UserRole.PUBLIC_USER
                        
                        user = db_service.auth.create_user(
                            db,
                            username=reg_username,
                            email=reg_email,
                            password=reg_password,
                            role=role,
                            first_name=reg_first_name,
                            last_name=reg_last_name,
                            organization=reg_organization,
                            phone=reg_phone
                        )
                        
                        if user:
                            st.success("Account created successfully! You can now sign in.")
                            st.balloons()
                        else:
                            st.error("Username or email already exists. Please try different credentials.")
                        
                        db.close()

def logout():
    """Log out the current user"""
    # Clear session state
    for key in ['authenticated', 'user', 'user_role', 'session_token']:
        if key in st.session_state:
            del st.session_state[key]
    
    st.success("You have been logged out successfully.")
    st.rerun()

def show_user_profile():
    """Display user profile information"""
    if not st.session_state.authenticated:
        return
    
    user = st.session_state.user
    
    st.subheader("👤 User Profile")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info(f"""
        **Account Information**
        - Username: {user['username']}
        - Email: {user['email']}
        - Role: {user['role'].replace('_', ' ').title()}
        - Organization: {user.get('organization', 'Not specified')}
        """)
    
    with col2:
        st.info(f"""
        **Personal Information**
        - Name: {user.get('first_name', '')} {user.get('last_name', '')}
        - Account Type: {user['role'].replace('_', ' ').title()}
        """)
    
    if st.button("🚪 Logout", type="secondary"):
        logout()

def show_user_preferences():
    """Display and manage user preferences"""
    if not st.session_state.authenticated:
        return
    
    user_id = st.session_state.user['id']
    
    st.subheader("⚙️ Alert Preferences")
    
    # Get current preferences
    db = next(get_db())
    
    try:
        current_prefs = db_service.preferences.get_user_alert_preferences(db, user_id)
        
        # Alert type preferences
        st.markdown("**Configure Alert Types & Severity**")
        
        alert_types = ["Air Quality", "Disease Outbreak", "Hospital Capacity", "Trend Alert"]
        severity_levels = ["LOW", "MEDIUM", "HIGH"]
        
        with st.form("alert_preferences"):
            preferences = {}
            
            for alert_type in alert_types:
                st.markdown(f"**{alert_type}**")
                col1, col2, col3, col4 = st.columns(4)
                
                # Find existing preference
                existing_pref = next((p for p in current_prefs if p.alert_type.value == alert_type), None)
                
                with col1:
                    min_severity = st.selectbox(
                        f"Min Severity",
                        severity_levels,
                        index=severity_levels.index(existing_pref.severity_threshold.value) if existing_pref else 1,
                        key=f"severity_{alert_type}"
                    )
                
                with col2:
                    email_enabled = st.checkbox(
                        "Email",
                        value=existing_pref.email_enabled if existing_pref else True,
                        key=f"email_{alert_type}"
                    )
                
                with col3:
                    sms_enabled = st.checkbox(
                        "SMS",
                        value=existing_pref.sms_enabled if existing_pref else False,
                        key=f"sms_{alert_type}"
                    )
                
                with col4:
                    push_enabled = st.checkbox(
                        "In-App",
                        value=existing_pref.push_enabled if existing_pref else True,
                        key=f"push_{alert_type}"
                    )
                
                preferences[alert_type] = {
                    'severity': min_severity,
                    'email': email_enabled,
                    'sms': sms_enabled,
                    'push': push_enabled
                }
            
            if st.form_submit_button("Save Preferences"):
                try:
                    # Save preferences to database
                    from database_models import AlertSeverity, AlertType
                    
                    for alert_type_str, prefs in preferences.items():
                        alert_type_enum = getattr(AlertType, alert_type_str.upper().replace(" ", "_"))
                        severity_enum = getattr(AlertSeverity, prefs['severity'])
                        
                        db_service.preferences.set_alert_preference(
                            db,
                            user_id,
                            alert_type_enum,
                            severity_enum,
                            prefs['email'],
                            prefs['sms'],
                            prefs['push']
                        )
                    
                    st.success("Alert preferences saved successfully!")
                except Exception as e:
                    st.error(f"Error saving preferences: {e}")
        
        # Location preferences
        st.markdown("---")
        st.markdown("**Location Preferences**")
        
        # Get user locations
        user_locations = db_service.preferences.get_user_locations(db, user_id)
        all_locations = db_service.health_data.get_locations(db)
        
        if user_locations:
            st.write("**Your Saved Locations:**")
            for loc in user_locations:
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"📍 {loc.name}")
                with col2:
                    if st.button("Remove", key=f"remove_{loc.id}"):
                        # TODO: Implement location removal
                        st.info("Location removal feature coming soon!")
        
        # Add new location
        st.write("**Add New Location:**")
        available_locations = [loc for loc in all_locations if loc not in user_locations]
        
        if available_locations:
            selected_location = st.selectbox(
                "Select Location",
                options=[None] + available_locations,
                format_func=lambda x: "Choose a location..." if x is None else x.name
            )
            
            if selected_location and st.button("Add Location"):
                try:
                    db_service.preferences.add_user_location(
                        db, user_id, str(selected_location.id)
                    )
                    st.success(f"Added {selected_location.name} to your locations!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error adding location: {e}")
        
    finally:
        db.close()

def check_authentication():
    """Check if user is authenticated and validate session"""
    init_session_state()
    
    # If user claims to be authenticated, validate session token
    if st.session_state.authenticated and st.session_state.session_token:
        db = next(get_db())
        try:
            user = db_service.auth.validate_session(db, st.session_state.session_token)
            if not user:
                # Invalid session, log out
                st.session_state.authenticated = False
                st.session_state.user = None
                st.session_state.user_role = None
                st.session_state.session_token = None
        finally:
            db.close()
    
    return st.session_state.authenticated

def require_role(required_role: UserRole):
    """Decorator to require specific user role"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            if not st.session_state.authenticated:
                st.error("Please log in to access this feature.")
                return
            
            user_role = st.session_state.user_role
            
            # Check role hierarchy: ADMIN > HEALTH_AUTHORITY > PUBLIC_USER
            role_hierarchy = {
                UserRole.PUBLIC_USER: 1,
                UserRole.HEALTH_AUTHORITY: 2,
                UserRole.ADMIN: 3
            }
            
            if role_hierarchy.get(user_role, 0) < role_hierarchy.get(required_role, 999):
                st.error(f"Access denied. This feature requires {required_role.value.replace('_', ' ').title()} privileges.")
                return
            
            return func(*args, **kwargs)
        return wrapper
    return decorator

def show_sidebar_user_info():
    """Show user information in sidebar"""
    if st.session_state.authenticated:
        user = st.session_state.user
        
        st.sidebar.markdown("---")
        st.sidebar.markdown("**👤 User Info**")
        st.sidebar.write(f"Welcome, {user.get('first_name', user['username'])}!")
        st.sidebar.write(f"Role: {user['role'].replace('_', ' ').title()}")
        
        if st.sidebar.button("Profile & Settings"):
            st.session_state.show_profile = True
        
        if st.sidebar.button("Logout"):
            logout()
    else:
        st.sidebar.markdown("---")
        st.sidebar.info("Please log in to access personalized features")

# Authentication utilities
def get_current_user() -> Optional[Dict[str, Any]]:
    """Get current authenticated user"""
    if st.session_state.authenticated:
        return st.session_state.user
    return None

def get_current_user_id() -> Optional[str]:
    """Get current user ID"""
    user = get_current_user()
    return user['id'] if user else None

def has_role(role: UserRole) -> bool:
    """Check if current user has specific role or higher"""
    if not st.session_state.authenticated:
        return False
    
    user_role = st.session_state.user_role
    
    role_hierarchy = {
        UserRole.PUBLIC_USER: 1,
        UserRole.HEALTH_AUTHORITY: 2,
        UserRole.ADMIN: 3
    }
    
    return role_hierarchy.get(user_role, 0) >= role_hierarchy.get(role, 999)