#include <random>
#include "math.h"
typedef enum _user_event_id
{
    default_event = 0,  // compiler defaults
    null_event = 1000,  // id offset
    on_reset = 1001,    // prm<1>: skip
    on_step = 1002,     // prm<1>: skip
}user_event_id;


typedef enum _user_request_id
{
    default_request = 0,        // compiler defaults
    null_request = 2000,        // id offset
    add_act_noise = 2001,       // prms<2>: min max
    randomize_bodypos = 2002,   // prms<6>: xmin ymin zmin xmax ymax zmax
    copy_bodypos = 2003,        // prms<1>: body_id x_offset y_offset z_offset
    randomize_sitepos = 2004,   // prms<6>: xmin ymin zmin xmax ymax zmax
    randomize_bodyeuler = 2005, // prms<6>: xmin ymin zmin xmax ymax zmax
    randomize_jnt = 2006,       // prms<6>: min max (only for hinge and slider)
}user_request_id;


template <class RealType>
static void normalDist(RealType* data, size_t count)
{
    std::minstd_rand gen;
    std::normal_distribution<RealType> d;
    for (size_t i = 0; i < count; i++)
        data[i] = d(gen);
}


template <class RealType>
static void uniformDist(RealType* data, double min, double max, size_t count)
{
    static std::default_random_engine generator;
    static std::uniform_real_distribution<RealType> distribution(0.0, 1.0);
    for (size_t i = 0; i< count; ++i)
        data[i] = min + (max - min)*distribution(generator);
}

// process event. Return true is the event is active
bool user_event(mjModel *m, mjData *d, int event_id, double* prms)
{
    int skip = prms[0];
    int step_cnt = (int)(d->time / m->opt.timestep);

    // process events
    switch (event_id)
    {
    case default_event:
        break;

    case on_reset:
        if (d->time == 0)
            return true;
        break;

    case on_step:
        if (!skip || (step_cnt%skip == 0))
            return true;
        break;

    default:
        // printf("\nWARNING: Unrecognised event: %d", event_id);
        return false;
    }
    return false;
}

//Convert Euler Angles to Quaternions
void euler2quat(double *quat, double* euler)
{
    double a[3] = {euler[2] / 2.0, -euler[1] / 2.0, euler[0] / 2.0};
    double s[3] = {sin(a[0]), sin(a[1]), sin(a[2])};



    double c[3] = {cos(a[0]), cos(a[1]), cos(a[2]) };
    double cc = c[0] * c[2];
    double cs = c[0] * s[2];
    double sc = s[0] * c[2];
    double ss = s[0] * s[2];

    quat[0] = c[1] * cc + s[1] * ss;
    quat[3] = c[1] * sc - s[1] * cs;
    quat[2] = -(c[1] * ss + s[1] * cc);
    quat[1] = c[1] * cs - s[1] * sc;

    double res = sqrt(quat[0]*quat[0] + quat[1]*quat[1] + quat[2]*quat[2] + quat[3]*quat[3]);
    quat[0] = quat[0]/res;
    quat[1] = quat[1]/res;
    quat[2] = quat[2]/res;
    quat[3] = quat[3]/res;
}

// process user requests
void user_requests(mjModel *m, mjData *d, int request_idx, int request_id, mjtNum* prms)
{
    int i_user = 0, reset_skip = 0, step_skip = 0, qpos_adr = 0, jnt_type = 0;

    // process requests
    //printf("\nrequest id: %d", request_id);
    switch (request_id)
    {
    case default_request:
        break;
    case null_request:
        break;
        // inject noise
    case add_act_noise:
        mjtNum noise;
        uniformDist(&noise, prms[0], prms[1], 1);
        d->ctrl[request_idx] +=noise;
        //printf("injecting act noise");
        break;

        // randomize body pos
    case randomize_bodypos:
        //printf("\nrequest id: %d ", request_id);
        //printf("Randomizing pos, %d", request_idx);
        //printf("prms: %f, %f, %f, %f, %f, %f", prms[0], prms[1], prms[2], prms[3], prms[4], prms[5]);
        //printf("\nBody pos: %f %f %f", m->body_pos[3 * request_idx], m->body_pos[3 * request_idx + 1], m->body_pos[3 * request_idx + 2]);
        uniformDist(m->body_pos + 3 * request_idx, prms[0], prms[3], 1);
        uniformDist(m->body_pos + 3 * request_idx + 1, prms[1], prms[4], 1);
        uniformDist(m->body_pos + 3 * request_idx + 2, prms[2], prms[5], 1);
        //printf("\nBody pos: %f %f %f", m->body_pos[3 * request_idx], m->body_pos[3 * request_idx + 1], m->body_pos[3 * request_idx + 2]);
        break;
    case randomize_bodyeuler:
        double euler[3];
        uniformDist(euler, prms[0], prms[3], 1);
        uniformDist(euler + 1, prms[1], prms[4], 1);
        uniformDist(euler + 2, prms[2], prms[5], 1);
        euler2quat(m->body_quat + 4*request_idx, euler);
        break;
    case copy_bodypos:
        m->body_pos[3 * request_idx] = m->body_pos[3 * (int)prms[0]] + prms[1];
        m->body_pos[3 * request_idx + 1] = m->body_pos[3 * (int)prms[0] + 1]+prms[2];
        m->body_pos[3 * request_idx + 2] = m->body_pos[3 * (int)prms[0] + 2]+prms[3];
        break;
    case randomize_sitepos:
        uniformDist(m->site_pos + 3 * request_idx, prms[0], prms[3], 1);
        uniformDist(m->site_pos + 3 * request_idx + 1, prms[1], prms[4], 1);
        uniformDist(m->site_pos + 3 * request_idx + 2, prms[2], prms[5], 1);
        break;
    case randomize_jnt:
        qpos_adr = m->jnt_qposadr[request_idx];
        jnt_type = m->jnt_type[request_idx];
        if( (jnt_type==(mjtJoint)mjJNT_SLIDE) || (jnt_type==(mjtJoint)mjJNT_SLIDE) )
        {   mjtNum qpos = 0.0;
            uniformDist(&qpos, prms[0], prms[1], 1);
            d->qpos[qpos_adr] = m->qpos0[qpos_adr] + qpos;
        }
        else
            printf("\nERROR: Request only suported for hinge and slide joint");
        break;
    default:
        printf("\nWARNING: Unknown command %d", request_id);
        break;
    }
}

// user demands
void user_step(mjModel* m, mjData* d)
{
    int event_id;
    mjtNum* event_prms;
    int request_id;
    mjtNum* request_prms;

    // Close logs on reset and save the xml used as well
    if ((user_event(m, d, on_reset, NULL))&&(strcmp(opt->logFile,"none")!=0)&&saveLogs)
    {
        saveLogs = false; // stop savnig logs
        write_logs(m, d, opt->logFile, true);
        printf("\tLogs Saved: %s%s.mjl\n", opt->logFile, logTimestr);
        char error[1000] = "Could not save model";        
        char name[100];
        sprintf(name, "%s_%s.xml", opt->logFile, logTimestr);
        mj_saveLastXML(name, m, error, 1000);
        printf("\tModel saved: %s%s.xml\n", opt->logFile, logTimestr);
    }


    // process actuator requests //evnt, e0, cmd, c0....
    if (m->nuser_actuator>3)
        for (int iu = 0; iu < m->nu; iu++)
        {
            event_id = (int)m->actuator_user[m->nuser_actuator*iu];
            event_prms = m->actuator_user + m->nuser_actuator*iu + 1;
            if (user_event(m, d, event_id, event_prms))
            {
                request_id = (int)m->actuator_user[m->nuser_actuator*iu + 2];
                request_prms = m->actuator_user + m->nuser_actuator*iu + 3;
                user_requests(m, d, iu, request_id, request_prms);
            }
        }

    // process body requests
    if (m->nuser_body>3) //evnt, e0, cmd, c0....
        for (int ib = 0; ib < m->nbody; ib++)
        {
            event_id = (int)m->body_user[m->nuser_body*ib];
            event_prms = m->body_user + m->nuser_body*ib + 1;
            if (user_event(m, d, event_id, event_prms))
            {
                request_id = (int)m->body_user[m->nuser_body*ib + 2];
                request_prms = m->body_user + m->nuser_body*ib + 3;
                user_requests(m, d, ib, request_id, request_prms);
            }
        }

    // process site requests
    if (m->nuser_site>3) //evnt, e0, cmd, c0....
        for (int is = 0; is < m->nsite; is++)
        {
            event_id = (int)m->site_user[m->nuser_site*is];
            event_prms = m->site_user + m->nuser_site*is + 1;
            if (user_event(m, d, event_id, event_prms))
            {
                request_id = (int)m->site_user[m->nuser_site*is + 2];
                request_prms = m->site_user + m->nuser_site*is + 3;
                user_requests(m, d, is, request_id, request_prms);
            }
        }
    
    // process joints
    if (m->nuser_jnt>3) //evnt, e0, cmd, c0....
        for (int ij = 0; ij < m->njnt; ij++)
        {
            event_id = (int)m->jnt_user[m->nuser_jnt*ij];
            event_prms = m->jnt_user + m->nuser_jnt*ij + 1;
            if (user_event(m, d, event_id, event_prms))
            {
                request_id = (int)m->jnt_user[m->nuser_jnt*ij + 2];
                request_prms = m->jnt_user + m->nuser_jnt*ij + 3;
                user_requests(m, d, ij, request_id, request_prms);
            }
        }   
}